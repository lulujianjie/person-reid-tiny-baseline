import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes):    # modified by gu
    feat_dim = 2048

    if 'triplet' in cfg.LOSS_TYPE:
        triplet = TripletLoss(cfg.MARGIN, cfg.HARD_FACTOR)  # triplet loss

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'softmax' in cfg.LOSS_TYPE:
        if cfg.LOSS_LABELSMOOTH == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
            print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.LOSS_TYPE == 'triplet+softmax+center':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if cfg.LOSS_LABELSMOOTH == 'on':
                return cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return cfg.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        elif cfg.LOSS_TYPE == 'softmax+center':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if cfg.LOSS_LABELSMOOTH == 'on':
                return cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return cfg.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        elif cfg.LOSS_TYPE == 'triplet+softmax':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if cfg.LOSS_LABELSMOOTH == 'on':
                return cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0]
            else:
                return cfg.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0]
        elif cfg.LOSS_TYPE == 'softmax':
            if cfg.LOSS_LABELSMOOTH == 'on':
                return xent(score, target)
            else:
                return F.cross_entropy(score, target)
        else:
            print('unexpected loss type')

    return loss_func, center_criterion