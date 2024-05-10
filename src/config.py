import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 512
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}

# OPTIMIZER_PARAMS = {
#     'type': 'AdamW',
#     'lr': 0.001,            # 학습률
#     'weight_decay': 0.01,   # 가중치 감쇠 비율
#     'betas': (0.9, 0.999),  # 모멘텀 파라미터
#     'eps': 1e-8,            # 수치 안정성을 위한 작은 숫자
#     'amsgrad': False        # AMSGrad 알고리즘 비활성화
# }

# SCHEDULER_PARAMS = {
#     'type': 'ReduceLROnPlateau',
#     'mode': 'min',        # 또는 'max', 모니터링 지표에 따라 결정
#     #'min' = 손실이 작아질 때 더 나은 성능을 의미함
#     #'max' = 정확도가 커질 때 더 나은 성능을 의미함
#     'factor': 0.1,        # 학습률을 감소시키는 배수
#     'patience': 5,        # 개선이 없을 때 대기할 에포크 수
#     'threshold': 0.0001,  # 개선이 있는 것으로 간주되는 최소 변화량
#     'verbose': True,      # 학습률이 변경될 때 메시지 출력 여부
# }
# # 'threshold_mode'=rel, 'eps'=0, min_lr=0, eps=1e-8

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
# MODEL_NAME          = 'alexnet'
# MODEL_NAME          = 'resnet18'
# MODEL_NAME          = 'resnet50'
MODEL_NAME          = 'MyNetwork'


# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
