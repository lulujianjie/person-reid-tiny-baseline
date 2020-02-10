import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.BASE_LR
        weight_decay = cfg.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.BASE_LR * cfg.BIAS_LR_FACTOR
            weight_decay = cfg.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.OPTIMIZER == 'SGD':
        optimizer = getattr(torch.optim, cfg.OPTIMIZER)(params, momentum=cfg.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.OPTIMIZER)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.CENTER_LR)

    return optimizer, optimizer_center
