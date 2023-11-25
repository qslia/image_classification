import glob
import os
import pandas as pd
from dataset import FacesData
from sklearn.model_selection import train_test_split
from default import DEVICE, FACE_KEYPOINTS_TEST_DIR, \
    FACE_KEYPOINTS_TRAIN_DIR, FACE_KEYPOINTS_ROOT_DIR
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch
import numpy as np


def get_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = FacesData(train.reset_index(drop=True))
    test_dataset = FacesData(test.reset_index(drop=True))
    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, test_loader, train_dataset, test_dataset


def get_model():
    model = torchvision.models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, 3),
        nn.MaxPool2d(2),
        nn.Flatten()
    )
    model.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 136),
        nn.Sigmoid()
    )
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model.to(DEVICE), criterion, optimizer


def train_batch(img, kps, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(DEVICE))
    loss = criterion(_kps, kps.to(DEVICE))
    loss.backward()
    optimizer.step()
    return loss


def validate_batch(img, kps, model, criterion):
    model.eval()
    with torch.no_grad():
        _kps = model(img.to(DEVICE))
    loss = criterion(_kps, kps.to(DEVICE))
    return _kps, loss


if __name__ == '__main__':
    train_img_paths = glob.glob(os.path.join(
        FACE_KEYPOINTS_TRAIN_DIR, '*.jpg'))
    df = pd.read_csv(os.path.join(FACE_KEYPOINTS_ROOT_DIR,
                                  'training_frames_keypoints.csv'))
    train_loader, test_loader, _, _ = get_data(df)
    model, criterion, optimizer = get_model()
    
    train_loss, test_loss = [], []
    n_epochs = 50
    for epoch in range(n_epochs):
        print(f" epoch {epoch + 1} : 50")
        
        epoch_train_loss, epoch_test_loss = 0, 0
        
        for ix, (img, kps) in enumerate(train_loader):
            loss = train_batch(img, kps, model, optimizer, criterion)
            epoch_train_loss += loss.item()
        epoch_train_loss /= (ix + 1)
        
        for ix, (img, kps) in enumerate(test_loader):
            kps, loss = validate_batch(img, kps, model, criterion)
            epoch_test_loss += loss.item()
        epoch_test_loss /= (ix + 1)
        
        train_loss.append(epoch_train_loss)
        test_loss.append(epoch_test_loss)
    torch.save(model.state_dict(), 'face_keypoints_vgg16.pth')
    np.savez('face_keypoints', train_loss=train_loss, test_loss=test_loss)

            
        
            
        
