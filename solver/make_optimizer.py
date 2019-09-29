import torch

def make_optimizer(Cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = Cfg.BASE_LR
        weight_decay = Cfg.WEIGHT_DECAY
        # if "bias" in key:
        #     lr = Cfg.BASE_LR * Cfg.BIAS_LR_FACTOR
        #     weight_decay = Cfg.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if Cfg.OPTIMIZER == 'SGD':
        optimizer = getattr(torch.optim, Cfg.OPTIMIZER)(params, momentum=Cfg.MOMENTUM)

    else:
        optimizer = getattr(torch.optim, Cfg.OPTIMIZER)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=Cfg.CENTER_LR)

    return optimizer, optimizer_center