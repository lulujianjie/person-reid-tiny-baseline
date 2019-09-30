import os
import sys
from config import Config
from torch.backends import cudnn

sys.path.append('.')
from datasets import make_dataloader
from processor import do_inference
from model import make_model
from utils.logger import setup_logger

if __name__ == "__main__":
    Cfg = Config()
    log_dir = Cfg.LOG_DIR
    logger = setup_logger('{}.test'.format(Cfg.PROJECT_NAME), log_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_dataloader(Cfg)
    model = make_model(Cfg, num_classes)
    model.load_param(Cfg.WEIGHT)

    do_inference(Cfg,
        model,
        val_loader,
        num_query)
