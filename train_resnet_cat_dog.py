import torchvision
import torch.nn as nn
import torch
from default import DEVICE, CAT_DOG_TRAIN_DIR, CAT_DOG_TEST_DIR
from dataset import cat_dog
import numpy as np


def get_model():
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(DEVICE), loss_fn, optimizer


def train_batch(x, y, model, opt, loss_fn):
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
    train = cat_dog(CAT_DOG_TRAIN_DIR, DEVICE)
    val = cat_dog(CAT_DOG_TEST_DIR, DEVICE)
    train_dl = torch.utils.data.DataLoader(train, batch_size=32,
                                           shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val, batch_size=32,
                                         shuffle=True, drop_last=True)
    return train_dl, val_dl


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
        train_epoch_loss = np.array(train_epoch_losses).mean()

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

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)
        
    np.savez('resnet18_data',
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies)