import os
from torch.backends import cudnn

from config import Config
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss
from processor import do_train


if __name__ == '__main__':
    Cfg = Config()
    logger = setup_logger('{}'.format(Cfg.PROJECT_NAME), Cfg.LOG_DIR)
    logger.info("Running with config:\n{}".format(Cfg.PROJECT_NAME))

    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark = True
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    train_loader, val_loader, num_query, num_classes = make_dataloader(Cfg)
    model = make_model(Cfg, num_class=num_classes)

    loss_func,center_criterion = make_loss(Cfg, num_classes=num_classes)

    optimizer,optimizer_center = make_optimizer(Cfg, model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, Cfg.STEPS, Cfg.GAMMA,
                                  Cfg.WARMUP_FACTOR,
                                  Cfg.WARMUP_EPOCHS, Cfg.WARMUP_METHOD)

    do_train(
            Cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,  # modify for using self trained model
            loss_func,
            num_query
        )