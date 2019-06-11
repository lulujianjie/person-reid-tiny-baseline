class Config():
    def __init__(self):
        self.PROJECT_NAME = 'baseline-person-reid'
        self.LOG_DIR = "./log/" #log dir and saved model dir
        self.DATA_DIR = "/xxx/datasets/Market-1501-v15.09.15/"
        self.DEVICE_ID = "4"

        self.LOG_PERIOD = 50  # iteration of display training log
        self.CHECKPOINT_PERIOD = 5  # save model period
        self.EVAL_PERIOD = 5  # validation period
        self.MAX_EPOCHS = 120

        #data loader
        self.DATALOADER_NUM_WORKERS = 8
        self.SAMPLER = 'softmax'
        self.BATCHSIZE = 64
        self.NUM_IMG_PER_ID = 4

        #model
        self.INPUT_SIZE = [128, 64] #HxW
        self.MODEL_NAME = "resnet50"
        self.LAST_STRIDE = 1
        self.PRETRAIN_PATH = "/xxx/pretrained_model/resnet50-19c8e357.pth"
        self.MODEL_NECK = 'bnneck'#'bnneck'
        self.NECK_FEAT = "after"
        self.PRETRAIN_CHOICE = 'imagenet'

        #loss
        self.LOSS_TYPE = 'triplet+center+softmax'
        self.LOSS_LABELSMOOTH = 'on'

        #solver

        self.OPTIMIZER = 'Adam'
        self.BASE_LR = 0.00035

        self.CE_LOSS_WEIGHT = 1.0
        self.TRIPLET_LOSS_WEIGHT = 1.0
        self.CENTER_LOSS_WEIGHT = 0.0005

        self.WEIGHT_DECAY = 0.0005
        self.BIAS_LR_FACTOR = 2
        self.WEIGHT_DECAY_BIAS = 0.0
        self.MOMENTUM = 0.9
        self.CENTER_LR = 0.5
        self.MARGIN = 0.3

        self.STEPS = [30,50,70]
        self.GAMMA = 0.1
        self.WARMUP_FACTOR = 0.01
        self.WARMUP_EPOCHS = 10
        self.WARMUP_METHOD = "linear" #option: 'linear','constant'

        #test
        self.TEST_IMS_PER_BATCH = 128
        self.FEAT_NORM = "yes"
        self.WEIGHT = '/xxx/pretrained_model/resnet50_person_reid_128x64.pth'
        self.DIST_MAT = self.LOG_DIR+"dist_mat.npy"
        self.VIDS = self.LOG_DIR+"vids.npy"
        self.CAMIDS = self.LOG_DIR+"camids.npy"
        self.TEST_METHOD = 'cosine'