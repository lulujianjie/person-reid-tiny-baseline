import logging
import time
import torch
import torch.nn as nn

from utils.meter import AverageMeter
from utils.metrics import R1_mAP
import numpy as np

def do_train(Cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
             scheduler,
        loss_fn,
        num_query):
    log_period = Cfg.LOG_PERIOD
    checkpoint_period = Cfg.CHECKPOINT_PERIOD
    eval_period = Cfg.EVAL_PERIOD
    output_dir = Cfg.LOG_DIR

    device = "cuda"
    epochs = Cfg.MAX_EPOCHS

    logger = logging.getLogger('{}.train'.format(Cfg.PROJECT_NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=Cfg.FEAT_NORM)
    #train
    for epoch in range(1, epochs+1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        model.train()
        for iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            score, feat = model(img, target)
            loss = loss_fn(score, feat, target)

            loss.backward()
            optimizer.step()
            if 'center' in Cfg.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / Cfg.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(),img.shape[0])
            acc_meter.update(acc,1)

            if (iter+1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (iter+1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        scheduler.step()
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), output_dir+Cfg.MODEL_NAME+'_{}.pth'.format(epoch))

        if epoch % eval_period == 0:
            model.eval()
            for iter, (img, vid, camid) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    feat = model(img)
                    evaluator.update((feat, vid, camid))

            cmc, mAP, _, _, _,_ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

def do_inference(Cfg,
        model,
        val_loader,
        num_query):

    device = "cuda"
    logger = logging.getLogger('{}.test'.format(Cfg.PROJECT_NAME))
    logger.info("Enter inferencing")
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=Cfg.FEAT_NORM, method=Cfg.TEST_METHOD)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for iter, (img, vid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)
            evaluator.update((feat, vid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, vids, camids, feats = evaluator.compute()

    np.save(Cfg.DIST_MAT, distmat)
    np.save(Cfg.VIDS, vids)
    np.save(Cfg.CAMIDS, camids)
    np.save(Cfg.IMG_PATH, img_path_list[num_query:])
    torch.save(feats, Cfg.FEATS)
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
