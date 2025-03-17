import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='HPI configuration')
parser.add_argument('--SAVE_LOG_DIR',type=str,default='./log')
parser.add_argument('--SAVE_LOG' ,type=str2bool,default= True)
parser.add_argument('--SAVE_RESULT_DIR',type=str,default='')

parser.add_argument('--PRETRAINED_MODEL_3D',type=str,default='')
parser.add_argument('--PRETRAINED_MODEL_INT',type=str,default='')
parser.add_argument('--PRETRAINED_MODEL_PROP',type=str,default='')
parser.add_argument('--PRETRAINED_MODEL_DEEPLABV3PLUS',type=str,default="")

parser.add_argument('--TRAIN_INT_LR',type=float,default = 1e-4)
parser.add_argument('--TRAIN_INT_LR_STEP',help='Step at which the learning rate decays', nargs="*", default=[], type=int)
parser.add_argument('--TRAIN_INT_LR_GAMMA',help='Step at which the learning rate decays', default=0.1, type=float)
# parser.add_argument('--TRAIN_INT_POWER',type=float,default = 0.9)
# parser.add_argument('--TRAIN_INT_MOMENTUM',type=float,default = 0.9)
parser.add_argument('--TRAIN_INT_WEIGHT_DECAY',type=float,default = 1e-7)
parser.add_argument('--TRAIN_INT_BATCH_SIZE',type=int,default = 1) # 18 for single RTX2080
parser.add_argument('--TRAIN_INT_NUM_WORKER',type=int,default = 12) # 6
# parser.add_argument('--TRAIN_INT_TOTAL_STEPS',type=int,default = 20) # 1000
# parser.add_argument('--TRAIN_INT_ROUND_PER_STEP',type=int,default = 3) # 3
# parser.add_argument('--TRAIN_INT_EPO_PER_ROUND',type=int,default = 10) # 30
parser.add_argument('--TRAIN_INT_TOTAL_EPO',type=int,default = 500) # 80000
parser.add_argument('--TRAIN_INT_SAVE_MODEL_INTERVAL',type=int,default = 50)
parser.add_argument('--TRAIN_INT_SAVE_IMG_INTERVAL',type=int,default = 20)

parser.add_argument('--TRAIN_PROP_LR',type=float,default = 0.0001)
parser.add_argument('--TRAIN_PROP_LR_STEP',help='Step at which the learning rate decays', nargs="*", default=[], type=int)
parser.add_argument('--TRAIN_PROP_LR_GAMMA',help='Step at which the learning rate decays', default=0.1, type=float)
parser.add_argument('--TRAIN_PROP_WEIGHT_DECAY',type=float,default = 1e-7)
parser.add_argument('--TRAIN_PROP_BATCH_SIZE',type=int,default = 1) 
parser.add_argument('--TRAIN_PROP_NUM_WORKER',type=int,default = 2)
parser.add_argument('--TRAIN_PROP_TOTAL_EPO',type=int,default = 600) # 80000
parser.add_argument('--TRAIN_PROP_SAVE_MODEL_INTERVAL',type=int,default = 50)
parser.add_argument('--TRAIN_PROP_SAVE_IMG_INTERVAL',type=int,default = 20)


parser.add_argument('--TRAIN_OBJECT',type=str,default= "organ")
parser.add_argument('--TRAIN_DATASET_LENGTH',type=int,default= 1)

parser.add_argument('--DATA_ROOT',type=str,default= "./sample_data")
parser.add_argument('--DATA_TRAIN_LIST',type=str,default= "./data/train.txt")
parser.add_argument('--DATA_TEST_LIST',type=str,default= "./data/test.txt")
parser.add_argument('--DATA_VISUAL_LIST',type=str,default= "./data/visual.txt")


parser.add_argument('--DATA_INFO_FILE',type=str,default= "")

parser.add_argument('--DATA_RANDOMFLIP',type=float, default=0.5)
parser.add_argument('--DATA_RESCALE',type=int,default= 416)
parser.add_argument('--DATA_RANDOMCROP',type=int,default = 416)

parser.add_argument('--TRAIN_TOP_K_PERCENT_PIXELS',type=float,default=0.15)
parser.add_argument('--TRAIN_HARD_MINING_STEP',type=int,default=50000)

# cfg=parser.parse_args()
cfg=parser.parse_known_args()[0]
                
# if not torch.cuda.is_available():
#         raise ValueError('config.py: cuda is not avalable')
