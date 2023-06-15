import gc
import time
import torch
from tqdm import tqdm
from utils import visualize
from metric import init_loss_meters, update_losses, log_results

def train(model, train_dl, val_dl, epochs, display_every=200, save_model=True, save_fig=True):
    fixed_val_data = next(iter(val_dl))

    for e in range(1, epochs + 1):
        loss_meters = init_loss_meters()
        i = 0

        for data in tqdm(train_dl):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model.setup_input(data)

            model.optimize()

            update_losses(model, loss_meters, data["L"].shape[0])

            i += 1
            if i % display_every == 0:
                print(f"Epoch {e}/{epochs}: Iteration {i}/{len(train_dl)}")

                log_results(loss_meters)

                visualize(model, fixed_val_data, save_fig, e)

        if save_model:
            torch.save(model.state_dict(), f"{e}_{time.time()}.pt")
            torch.save(model.D.state_dict(), f"{e}_D_{time.time()}.pt")
            torch.save(model.G.state_dict(), f"{e}_G_{time.time()}.pt")

def test():
    import glob
    import numpy as np
    from torch.utils.data import DataLoader
    from fastai.data.external import untar_data, URLs
    from datasets import TrainingDataset, TestDataset
    from discriminator import PatchDiscriminator
    from generator import Generator
    from gan import MainModel

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
    d = PatchDiscriminator(1, 2)
    g = Generator(1, 2)

    m = MainModel(device, d, g)

    train(m, train_dl, val_dl, 10, 1, False, False)

if __name__ == "__main__":
    test()
