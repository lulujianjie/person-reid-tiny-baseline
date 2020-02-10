# coding=utf-8
"""
It is NOT RECOMMENDED for developers to modify the base class directly.
Instead, they should re derive a new configuration class in configs.py
"""
class DefaultConfig:
    """ Base configuration class for perparameter settings.
    All new configuration should be derived from this class.
    """

    def __init__(self):
        self.PROJECT_NAME = 'person-reid-tiny-baseline'  # project name
        self.LOG_DIR = "./log"  # log directory
        self.OUTPUT_DIR = "./output"  # saved model directory
        self.DEVICE_ID = "0"  # GPU IDs, i.e. "0,1,2" for multiple GPUs

        self.LOG_PERIOD = 50  # iteration of displaying training log
        self.CHECKPOINT_PERIOD = 5  # saving model period
        self.EVAL_PERIOD = 5  # validation period
        self.MAX_EPOCHS = 200  # max training epochs

        # data
        self.DATA_DIR = "/home/lujj/datasets/Market-1501-v15.09.15/"  # dataset path
        self.DATALOADER_NUM_WORKERS = 8  # number of dataloader workers
        self.SAMPLER = 'triplet'  # batch sampler, option: 'triplet','softmax'
        self.BATCH_SIZE = 64  # MxN, M: number of persons, N: number of images of per person
        self.NUM_IMG_PER_ID = 4  # N, number of images of per person

        # model
        self.INPUT_SIZE = [256, 128]  # HxW
        self.MODEL_NAME = "resnet50"  # backbone name, option: 'resnet50',
        self.LAST_STRIDE = 1  # the stride of the last layer of resnet50
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = "/xxx/pretrained_model/resnet50-19c8e357.pth"  # pretrained weight path

        # loss
        self.LOSS_TYPE = 'softmax'  # option: 'triplet+softmax','softmax+center','triplet+softmax+center'
        self.LOSS_LABELSMOOTH = 'on'  # using labelsmooth, option: 'on', 'off'
        self.COS_LAYER = False

        # solver
        self.OPTIMIZER = 'Adam'  # optimizer
        self.BASE_LR = 0.00035  # base learning rate

        self.CE_LOSS_WEIGHT = 1.0  # weight of softmax loss
        self.TRIPLET_LOSS_WEIGHT = 1.0  # weight of triplet loss
        self.CENTER_LOSS_WEIGHT = 0.0005  # weight of center loss

        self.HARD_FACTOR = 0.0 # harder example mining

        self.WEIGHT_DECAY = 0.0005
        self.BIAS_LR_FACTOR = 1.0
        self.WEIGHT_DECAY_BIAS = 0.0005
        self.MOMENTUM = 0.9
        self.CENTER_LR = 0.5  # learning rate for the weights of center loss
        self.MARGIN = 0.3  # triplet loss margin

        self.STEPS = [40, 70, 130]
        self.GAMMA = 0.1  # decay factor of learning rate
        self.WARMUP_FACTOR = 0.01
        self.WARMUP_EPOCHS = 10  # warm up epochs
        self.WARMUP_METHOD = "linear"  # option: 'linear','constant'

        # test
        self.TEST_IMS_PER_BATCH = 128
        self.FEAT_NORM = "yes"
        self.TEST_WEIGHT = './output/resnet50_175.pth'
        self.DIST_MAT = "dist_mat.npy"
        self.PIDS = "pids.npy"
        self.CAMIDS = "camids.npy"
        self.IMG_PATH = "imgpath.npy"
        self.Q_FEATS = "qfeats.pth"  # query feats
        self.G_FEATS = "gfeats.pth"  # gallery feats
        self.TEST_METHOD = 'cosine'
        self.FLIP_FEATS = 'off'  # using fliped feature for testing, option: 'on', 'off'
        self.RERANKING = False  # re-ranking
        self.QUERY_DIR = '/home/lujj/datasets/Market-1501-v15.09.15/query/'
