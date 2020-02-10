import os
from torch.backends import cudnn

from config import Config
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    cfg = Config()
    log_dir = cfg.LOG_DIR
    logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), log_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_classes)
    model.load_param(cfg.TEST_WEIGHT)

    do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
