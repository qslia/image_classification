from dataset import GenderAgeClass
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from model import ageGenderClassifier
import torch
from default import DEVICE
import pandas as pd
import time
import numpy as np


def get_data(trn_df, val_df):
    trn = GenderAgeClass(trn_df)
    val = GenderAgeClass(trn_df)
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
    return total_loss


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
    trn_df = pd.read_csv('fairface-labels-train.csv')
    val_df = pd.read_csv('fairface-labels-val.csv')
    train_loader, test_loader, _, _ = get_data(trn_df=trn_df, val_df=val_df)
    model, criterion, optimizer = get_model()
    # model.load_state_dict(torch.load('GenderAge_vgg16.pth'))

    val_gender_accuracies = []
    val_age_maes = []
    train_losses = []
    val_losses = []

    n_epochs = 5
    best_test_loss = 1000
    start = time.time()

    for epoch in range(n_epochs):
        epoch_train_loss_epoch, epoch_test_loss_epoch = 0, 0
        val_age_mae_epoch, val_gender_acc_epoch, counter = 0, 0, 0

        for ix, data in enumerate(train_loader):
            loss = train_batch(data, model, optimizer, criterion)
            epoch_train_loss_epoch += loss.item()

        for ix, data in enumerate(test_loader):
            loss, gender_acc, age_mae = validate_batch(data, model, criterion)
            epoch_test_loss_epoch += loss.item()
            val_age_mae_epoch += age_mae
            val_gender_acc_epoch += gender_acc
            counter += len(data[0])

        val_age_mae_epoch /= counter
        val_gender_acc_epoch /= counter

        epoch_train_loss_epoch /= len(train_loader)
        epoch_test_loss_epoch /= len(test_loader)

        elapsed = time.time() - start
        best_test_loss = min(best_test_loss, epoch_test_loss_epoch)

        print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(
            epoch+1, n_epochs, time.time()-start,
            (n_epochs-epoch)*(elapsed/(epoch+1))))
        info = f'''Epoch: {epoch+1:03d} \
                    \tTrain Loss: {epoch_train_loss_epoch:.3f}
                    \tTest: {epoch_test_loss_epoch:.3f}
                    \tBest Test Loss: {best_test_loss:.4f}'''
        info += f'\nGender Accuracy:\
                    {val_gender_acc_epoch*100:.2f}%\tAge MAE:\
                        {val_age_mae_epoch:.2f}\n'
        print(info)

        val_gender_accuracies.append(val_gender_acc_epoch.cpu().numpy())
        val_age_maes.append(val_age_mae_epoch.cpu().numpy())

    torch.save(model.state_dict(), 'GenderAge_vgg16.pth')
    np.savez('GenderAge_vgg16',
             val_gender_accuracies=val_gender_accuracies,
             val_age_maes=val_age_maes)
