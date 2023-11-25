import torch

CAT_DOG_TRAIN_DIR = 'dataset/training_set/training_set'
CAT_DOG_TEST_DIR = 'dataset/test_set/test_set'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

FACE_KEYPOINTS_TRAIN_DIR = 'data/training/'
FACE_KEYPOINTS_TEST_DIR = 'data/test'
FACE_KEYPOINTS_ROOT_DIR = 'data'

FAIRFACE_ROOT_DIR = 'dataset/fairface-img-margin025-trainval'