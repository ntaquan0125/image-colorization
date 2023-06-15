import cv2
import torch
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.resnet import resnet18
from utils import lab_to_rgb
from generator import pretrained_generator

def images_to_tensors(paths):
    first_img = Image.open(paths[0])
    tensor_img = 2 * (transforms.ToTensor()(first_img))[0] - 1
    tensor_imgs = tensor_img.view(1, 1, *tensor_img.shape)

    for path in paths[1:]:
        img = Image.open(path)
    
        tensor_img = 2 * (transforms.ToTensor()(img))[0] - 1
        tensor_img = tensor_img.view(1, 1, *tensor_img.shape)

        tensor_imgs = torch.cat((tensor_imgs, tensor_img), 0)

    return tensor_imgs

def predict(tensors):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = pretrained_generator(256, 1, 2, resnet18, -2)
    g.load_state_dict(torch.load("g.pt", map_location=device))

    return g(tensors)

def visualize(tensors, color_imgs):
    fig = plt.figure(figsize=(20, 10))

    for i in range(5 if tensors.shape[0] > 5 else tensors.shape[0]):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(tensors[i, 0].cpu(), cmap="gray")
        ax.axis("off")

        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(color_imgs[i])
        ax.axis("off")

    plt.show()

def save_imgs(imgs):
    rgb_imgs = np.zeros(imgs.shape)
    rgb_imgs[:, :, :, 0] = 255 * imgs[:, :, :, 2]
    rgb_imgs[:, :, :, 1] = 255 * imgs[:, :, :, 1]
    rgb_imgs[:, :, :, 2] = 255 * imgs[:, :, :, 0]

    for i in range(imgs.shape[0]):
        cv2.imwrite(f"{i}_color.png", rgb_imgs[i])

def extract_frames(path, limit_frames=-1):
    source = cv2.VideoCapture(path)
    idx = 0
    ret = True

    while ret and (idx < limit_frames or limit_frames == -1):
        ret, frame = source.read()
        idx += 1
        time.sleep(.1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_img = Image.fromarray(gray_frame, "L")
        gray_img.save(f"{idx}.png")

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

if __name__ == "__main__":
    tensors = images_to_tensors(["1.jpg", "2.jpg", "3.jpg"])
    print(tensors.shape)

    outs = predict(tensors)

    color_imgs = lab_to_rgb(tensors, outs.detach())

    visualize(tensors, color_imgs)

    save_imgs(color_imgs)

    extract_frames("v.mp4", 10)
