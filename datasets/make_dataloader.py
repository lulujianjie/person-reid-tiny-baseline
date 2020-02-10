from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .Market1501 import Market1501
from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        # T.RandomRotation(12, resample=Image.BICUBIC, expand=False, center=None),
        # T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0),
        #                T.RandomAffine(degrees=0, translate=None, scale=[0.8, 1.2], shear=15, \
        #                               resample=Image.BICUBIC, fillcolor=0)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, sh=0.4, mean=(0.4914, 0.4822, 0.4465))
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers = cfg.DATALOADER_NUM_WORKERS
    dataset = Market1501(data_dir=cfg.DATA_DIR, verbose=True)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if cfg.SAMPLER == 'triplet':
        print('using triplet sampler')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.BATCH_SIZE,
                                  num_workers=num_workers,
                                  sampler=RandomIdentitySampler(dataset.train, cfg.BATCH_SIZE, cfg.NUM_IMG_PER_ID),
                                  collate_fn=train_collate_fn  # customized batch sampler
                                  )
    elif cfg.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  sampler=None,
                                  collate_fn=train_collate_fn,  # customized batch sampler
                                  drop_last=True
                                  )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(val_set,
                            batch_size=cfg.TEST_IMS_PER_BATCH,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=val_collate_fn
                            )
    return train_loader, val_loader, len(dataset.query), num_classes
