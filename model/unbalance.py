import time
import numpy as np
from tqdm import tqdm
from metric import AverageMeter

def unbalance_generator(device, G, train_dl, val_dl, opt, criterion, epochs, save=False):
    for e in range(1, epochs + 1):
        train_loss_meter = AverageMeter()
        val_loss_meter = AverageMeter()

        G.train()
        for data in tqdm(train_dl):
            L, ab = data["L"].to(device), data["ab"].to(device)

            preds = G(L)

            loss = criterion(preds, ab)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_meter.update(loss.item(), L.shape[0])

        G.eval()
        for data in tqdm(val_dl):
            L, ab = data["L"].to(device), data["ab"].to(device)

            preds = G(L)

            loss = criterion(preds, ab)

            val_loss_meter.update(loss.item(), L.shape[0])

        print(f"Epoch {e}/{epochs}:")
        print(f"train_loss: {train_loss_meter.avg:.4f}, val_loss: {val_loss_meter.avg:.4f}")
        
        if save:
            torch.save(G.state_dict(), f"{e}_{time.time()}_res18-unet.pt")

def test():
    import glob
    import torch
    from torch.utils.data import DataLoader
    from fastai.data.external import untar_data, URLs
    from datasets import TrainingDataset, TestDataset
    from generator import Generator

    root = str(untar_data(URLs.COCO_SAMPLE)) + "/train_sample"

    paths = glob.glob(root + "/*.jpg")
    print(len(paths))

    np.random.seed(42)

    paths_subset = np.random.choice(paths, 3, replace=False)

    print(paths_subset)

    rand_idxs = np.random.permutation(3)
    train_idxs = rand_idxs[:2]
    val_idxs = rand_idxs[2:]

    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]

    train_ds = TrainingDataset(256, train_paths)
    val_ds = TestDataset(256, val_paths)

    train_dl = DataLoader(train_ds, 1, num_workers=1, pin_memory=True)
    val_dl = DataLoader(val_ds, 1, num_workers=1, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = Generator(1, 2).to(device)
    opt = torch.optim.Adam(g.parameters(), lr=1e-4)
    loss = torch.nn.L1Loss()

    unbalance_generator(device, g, train_dl, val_dl, opt, loss, 5)

if __name__ == "__main__":
    test()
