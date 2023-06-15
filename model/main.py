import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from fastai.data.external import untar_data, URLs
from datasets import TrainingDataset, TestDataset
from discriminator import PatchDiscriminator
from generator import Generator, pretrained_generator
from gan import MainModel
from train import train

if __name__ == "__main__":
    root = str(untar_data(URLs.COCO_SAMPLE)) + "/train_sample"

    paths = glob.glob(root + "/*.jpg")
    print(len(paths))

    np.random.seed(42)

    paths_subset = np.random.choice(paths, 18_500, replace=False)

    print(paths_subset)

    rand_idxs = np.random.permutation(18_500)
    train_idxs = rand_idxs[:18_000]
    val_idxs = rand_idxs[18_000:]

    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]

    train_ds = TrainingDataset(256, train_paths)
    val_ds = TestDataset(256, val_paths)

    train_dl = DataLoader(train_ds, 16, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, 16, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = PatchDiscriminator(1, 2)
    # g = Generator(1, 2)
    g = pretrained_generator(256, 1, 2, resnet18, -2)
    # g.load_state_dict(torch.load("path"))

    m = MainModel(device, d, g)

    train(m, train_dl, val_dl, 20, 200, False, False)
