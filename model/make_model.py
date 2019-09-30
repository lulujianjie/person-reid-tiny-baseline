import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.arcface import ArcCos

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model = 'fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model = 'fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std = 0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Backbone(nn.Module):
    def __init__(self, num_classes, Cfg):
        super(Backbone, self).__init__()
        last_stride = Cfg.LAST_STRIDE
        model_path = Cfg.PRETRAIN_PATH
        neck = Cfg.MODEL_NECK  # If train with BNNeck, options: 'bnneck' or 'no'
        neck_feat = Cfg.NECK_FEAT
        self.cos_layer = Cfg.COS_LAYER
        model_name = Cfg.MODEL_NAME
        pretrain_choice = Cfg.PRETRAIN_CHOICE
        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])

        else:
            print('unsupported backbone! only support resnet50, but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        if self.cos_layer == 'yes':
            print('using cosine layer')
            self.arcface = ArcCos(self.in_planes, self.num_classes, s=30.0, m=0.50)
        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)

        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes, momentum=0.1, affine=False)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self,x, label=None):#label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.neck =='no':
            feat = global_feat
        elif 'bnneck' in self.neck:
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer == 'yes':
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i[7:]].copy_(param_dict[i])

def make_model(Cfg, num_class):
    model = Backbone(num_class, Cfg)
    return model