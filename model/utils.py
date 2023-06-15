import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50
    ab = ab * 110.

    Lab = torch.cat([L, ab], 1).permute(0, 2, 3, 1).cpu().numpy()

    rgb_imgs = []
    for lab_img in Lab:
        rgb_img = lab2rgb(lab_img)
        rgb_imgs.append(rgb_img)

    return np.stack(rgb_imgs, 0)

def visualize(model, data, save=True, prefix=""):
    model.G.eval()

    with torch.no_grad():
        model.setup_input(data)
        model.forward()

    model.G.train()

    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L

    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)

    fig = plt.figure(figsize=(15, 8))
    for i in range(5 if L.shape[0] > 5 else L.shape[0]):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i, 0].cpu(), cmap="gray")
        ax.axis("off")

        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")

        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")

    plt.show(block=False)

    if save:
        fig.savefig(f"{prefix}_{time.time()}.png")

def init_weights(net, mean=.0, std=.01):
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and "Conv" in classname:
            torch.nn.init.normal_(m.weight.data, mean=mean, std=std)
        elif "BatchNorm2d" in classname:
            torch.nn.init.normal_(m.weight.data, 1., std)
            torch.nn.init.constant_(m.bias.data, mean)

    net.apply(init_func)

    return net

def test():
    from discriminator import PatchDiscriminator
    from generator import Generator
    from gan import MainModel

    d = PatchDiscriminator(1, 2)
    g = Generator(1, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = MainModel(device, d, g)
    m = init_weights(m)

    large_data = {"L": torch.rand(32, 1, 256, 256), "ab": torch.rand(32, 2, 256, 256)}
    small_data = {"L": torch.rand(3, 1, 256, 256), "ab": torch.rand(3, 2, 256, 256)}

    visualize(m, large_data, False)
    visualize(m, small_data, False)

if __name__ == "__main__":
    test()
