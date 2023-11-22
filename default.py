import torch

TRAIN_DIR = 'dataset/training_set/training_set'
TEST_DIR = 'dataset/test_set/test_set'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'