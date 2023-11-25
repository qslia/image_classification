import torch
import glob
from torchvision.transforms import transforms
import random
import cv2
import os
from copy import deepcopy
import numpy as np
from default import FACE_KEYPOINTS_TRAIN_DIR, DEVICE, FAIRFACE_ROOT_DIR


class cat_dog(torch.utils.data.Dataset):
    def __init__(self, folder, device):
        cats = glob.glob(folder + '/cats/*.jpg')
        dogs = glob.glob(folder + '/dogs/*.jpg')
        self.fpaths = cats[:500] + dogs[:500]
        self.device = device
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        random.seed(10)
        random.shuffle(self.fpaths)
        self.targets = [fpath.split('/')[-1].startswith('dog')
                        for fpath in self.fpaths]

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = cv2.imread(f)[:, :, ::-1]
        im = cv2.resize(im, (224, 224))
        im = torch.tensor(im / 255.)
        im = im.permute(2, 0, 1)
        im = self.normalize(im).float().to(self.device)
        target = torch.tensor([target]).float().to(self.device)
        return im, target


class FacesData(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        img_path = FACE_KEYPOINTS_TRAIN_DIR + self.df.iloc[ix, 0]
        img = cv2.imread(img_path) / 255.
        kp = deepcopy(self.df.iloc[ix, 1:].tolist())
        kp_x = (np.array(kp[0::2]) / img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2]) / img.shape[0]).tolist()
        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2)
        img = self.preprocess_input(img)
        return img, kp2

    def preprocess_input(self, img):
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2, 0, 1)
        img = self.normalize(img).float()
        return img.to(DEVICE)

    def load_img(self, ix):
        img_path = FACE_KEYPOINTS_TRAIN_DIR + self.df.iloc[ix, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = cv2.resize(img, (224, 224))
        return img


IMAGE_SIZE = 224


class GenderAgeClass(torch.utils.data.Dataset):
    def __init__(self, df, tfms=None):
        self.df = df
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        f = self.df.iloc[ix]
        file = os.path.join(FAIRFACE_ROOT_DIR, f.file)
        gen = f.gender == 'Female'
        age = f.age
        im = cv2.imread(file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, age, gen

    def preprocess_image(self, im):
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        im = torch.tensor(im).permute(2, 0, 1)
        im = self.normalize(im/255.)
        return im[None]

    def collate_fn(self, batch):
        'preprocess images, ages and genders'
        ims, ages, genders = [], [], []
        for im, age, gender in batch:
            im = self.preprocess_image(im)
            ims.append(im)
            ages.append(float(int(age)/80))
            genders.append(float(gender))
        ages, genders = [torch.tensor(x).to(DEVICE).float()
                         for x in [ages, genders]]
        ims = torch.cat(ims).to(DEVICE)
        return ims, ages, genders
