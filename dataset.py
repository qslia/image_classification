import torch
import glob
from torchvision.transforms import transforms
import random
import cv2


class cat_dog(torch.utils.data.Dataset):
    def __init__(self, folder, device):
        cats = glob.glob(folder + '/cats/*.jpg')
        dogs = glob.glob(folder + '/dogs/*.jpg')
        self.fpaths = cats[:500] + dogs[:500]
        self.device = device
        self.normalize = transforms.Normalize(mean=[0.485,
                                                    0.456, 0.406], std=[0.229, 0.224, 0.225])
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
