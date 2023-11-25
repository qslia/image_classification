from dataset import GenderAgeClass2
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from model import ageGenderClassifier
import torch
from default import DEVICE
import pandas as pd
import time
import numpy as np
from torch_snippets import Report


def get_data(trn_df, val_df):
    trn = GenderAgeClass2(trn_df)
    val = GenderAgeClass2(trn_df)
    train_loader = DataLoader(trn, batch_size=128, shuffle=True,
                              drop_last=True, collate_fn=trn.collate_fn)
    test_loader = DataLoader(val, batch_size=128, shuffle=True,
                             drop_last=True, collate_fn=val.collate_fn)
    return train_loader, test_loader, trn, val


def get_model():
    model = torchvision.models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    model.classifier = ageGenderClassifier()
    gender_criterion = nn.BCELoss()
    age_criterion = nn.L1Loss()
    loss_functions = gender_criterion, age_criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(DEVICE), loss_functions, optimizer


def train_batch(data, model, optimizer, criterion):
    model.train()
    ims, age, gender = data
    optimizer.zero_grad()
    pred_gender, pred_age = model(ims)
    gender_criterion, age_criterion = criterion
    gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    age_loss = age_criterion(pred_age.squeeze(), age)
    total_loss = gender_loss + age_loss
    optimizer.step()
    return total_loss, gender_loss, age_loss


def validate_batch(data, model, criterion):
    model.eval()
    img, age, gender = data
    with torch.no_grad():
        pred_gender, pred_age = model(img)
    gender_criterion, age_criterion = criterion
    gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    age_loss = age_criterion(pred_age.squeeze(), age)
    total_loss = gender_loss + age_loss
    pred_gender = (pred_gender > 0.5).squeeze()
    gender_acc = (pred_gender == gender).float().sum()
    age_mae = torch.abs(age - pred_age).float().sum()
    return total_loss, gender_acc, age_mae


if __name__ == '__main__':
    trn_df = pd.read_csv('dataset/fairface-labels-train.csv')
    val_df = pd.read_csv('dataset/fairface-labels-val.csv')
    train_loader, test_loader, _, _ = get_data(trn_df=trn_df, val_df=val_df)
    model, criterion, optimizer = get_model()

    n_epochs = 5
    log = Report(n_epochs)

    for epoch in range(n_epochs):
        
        N = len(train_loader)
        for ix, data in enumerate(train_loader):
            total_loss, gender_loss, age_loss = train_batch(data, 
                                            model, optimizer, criterion)
            log.record(epoch+(ix+1)/N, trn_loss=total_loss, end='\r')
            
        N = len(test_loader)
        for ix, data in enumerate(test_loader):
            total_loss, gender_acc, age_mae = validate_batch(data, \
                            model, criterion)
            gender_acc /= len(data[0])
            age_mae /= len(data[0])
            
            log.record(epoch+(ix+1)/N, val_loss=total_loss, \
                        val_gender_acc=gender_acc, \
                        val_age_mae=age_mae, end='\r')
        log.report_avgs(epoch+1)
    # log.plot_epochs() 
    log.save(filename='1.log', ignore_discard=False, ignore_expires=False)   
    torch.save(model.state_dict(), 'GenderAge_vgg16.pth')
