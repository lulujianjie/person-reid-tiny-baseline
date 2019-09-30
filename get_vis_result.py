import os
import sys
from config import Config
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from PIL import Image
sys.path.append('.')
from utils.logger import setup_logger
from model import make_model

import numpy as np
import cv2
from utils.metrics import cosine_similarity


def visualizer(test_img, camid, top_k = 10, img_size=[128,128]):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    for k in range(top_k):
        name = str(indices[0][k]).zfill(6)
        img = np.asarray(Image.open(img_path[indices[0][k]]).resize((img_size[1],img_size[0])))
        figure = np.hstack((figure, img))
        title=name
    figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)
    if not os.path.exists(Cfg.LOG_DIR+ "/results/"):
        print('need to create a new folder named results in {}'.format(Cfg.LOG_DIR))
    cv2.imwrite(Cfg.LOG_DIR+ "/results/{}-cam{}.png".format(test_img,camid),figure)

if __name__ == "__main__":
    Cfg = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark = True

    model = make_model(Cfg, 255)
    model.load_param(Cfg.WEIGHT)

    device = 'cuda'
    model = model.to(device)
    transform = T.Compose([
        T.Resize(Cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



    log_dir = Cfg.LOG_DIR
    logger = setup_logger('{}.test'.format(Cfg.PROJECT_NAME), log_dir)
    model.eval()
    for test_img in os.listdir(Cfg.QUERY_DIR):
        logger.info('Finding ID {} ...'.format(test_img))

        gallery_feats = torch.load(Cfg.LOG_DIR + 'feats.pth')
        img_path = np.load('./log/imgpath.npy')
        print(gallery_feats.shape, len(img_path))
        query_img = Image.open(Cfg.QUERY_DIR + test_img)
        input = torch.unsqueeze(transform(query_img), 0)
        input = input.to(device)
        with torch.no_grad():
            query_feat = model(input)

        dist_mat = cosine_similarity(query_feat, gallery_feats)
        indices = np.argsort(dist_mat, axis=1)
        visualizer(test_img, camid='mixed', top_k=10, img_size=Cfg.INPUT_SIZE)