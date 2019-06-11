import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader


from .Market1501 import Market1501
from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler

from PIL import Image
import numpy as np


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids
#collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


def make_dataloader(Cfg):
    train_transforms = T.Compose([
        T.Resize(Cfg.INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop(Cfg.INPUT_SIZE),
        #T.RandomRotation(10, resample=Image.BICUBIC, expand=False, center=None),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, sh=0.4, mean=(0.4914, 0.4822, 0.4465))
    ])

    val_transforms = T.Compose([
        T.Resize(Cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers = Cfg.DATALOADER_NUM_WORKERS
    dataset = Market1501(data_dir = Cfg.DATA_DIR, verbose = True)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if Cfg.SAMPLER == 'softmax':
        train_loader = DataLoader(train_set,
            batch_size = Cfg.BATCHSIZE,
            shuffle = False,
            num_workers = num_workers,
            sampler = RandomIdentitySampler(dataset.train, Cfg.BATCHSIZE, Cfg.NUM_IMG_PER_ID),
            collate_fn = train_collate_fn, #customized batch sampler
            drop_last = True
        )
    else:
        print('unsupported sampler! expected softmax but got {}'.format(Cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(val_set,
        batch_size=Cfg.TEST_IMS_PER_BATCH,
        shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes