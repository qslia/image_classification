import torch
import torchvision
import glob
from torchvision.transforms import transforms
import random
import cv2
from dataset import cat_dog
from default import TRAIN_DIR, TEST_DIR, DEVICE
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_model():
    model = torchvision.models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.classifier = nn.Sequential(nn.Flatten(),
                                     nn.Linear(512, 128),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(128, 1),
                                     nn.Sigmoid())
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(DEVICE), loss_fn, optimizer


def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()


def get_data():
    train_data = cat_dog(TRAIN_DIR, DEVICE)
    train_dl = DataLoader(train_data, batch_size=32,
                          shuffle=True, drop_last=True)
    val_data = cat_dog(TEST_DIR, DEVICE)
    val_dl = DataLoader(val_data, batch_size=32, shuffle=True, drop_last=True)

    return train_dl, val_dl


def plot_accuracy(epochs, train_accuracies, val_accuracies):
    epochs = np.arange(5) + 1
    plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    from matplotlib import ticker
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title('Training and validation accuracy with VGG16 \
        and 1K training data points')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0.95, 1)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100)
                               for x in plt.gca().get_yticks()])
    plt.legend()
    plt.grid('off')
    plt.show()


if __name__ == '__main__':
    train_dl, val_dl = get_data()
    model, loss_fn, optimizer = get_model()

    train_losses, train_accuracies, val_accuracies = [], [], []
    for epoch in range(5):
        print(f" epoch {epoch + 1} / 5")
        train_epoch_losses, train_epoch_accuracies, val_epoch_accuracies = [], [], []

        for ix, batch in enumerate(train_dl):
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)
        train_epoch_losses = np.array(train_epoch_losses).mean()

        for ix, batch in enumerate(train_dl):
            x, y = batch
            train_is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(train_is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        for ix, batch in enumerate(val_dl):
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            val_epoch_accuracies.extend(val_is_correct)
        val_epoch_accuracy = np.mean(val_epoch_accuracies)

        train_losses.append(train_epoch_losses)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)

    np.savez('vgg16_data',
             train_losses=train_losses,
             train_accuracies=train_accuracies,
             val_accuracies=val_accuracies)

    # data = np.load('vgg16_data.npz')
    # plot_accuracy(5, data['train_accuracies'], data['train_accuracies'])
