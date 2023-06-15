import torch
import torch.nn as nn
from torch import optim

class GANLoss(nn.Module):
    def __init__(self, real_label=1., fake_label=0.):
        super(GANLoss, self).__init__()

        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))

        self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, preds, is_real_target):
        if is_real_target:
            return self.real_label.expand_as(preds)

        return self.fake_label.expand_as(preds)

    def __call__(self, preds, is_real_target):
        labels = self.get_labels(preds, is_real_target)

        return self.loss(preds, labels)

class MainModel(nn.Module):
    def __init__(self, device, D, G, lr_G=2e-4, lr_D=2e-4, betas=(.5, .999), lambda_L1=100.):
        super(MainModel, self).__init__()

        self.device = device

        self.D = D.to(device)
        self.G = G.to(device)

        self.GANcriterion = GANLoss().to(device)
        self.L1criterion = nn.L1Loss()

        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_D, betas=betas)
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_G, betas=betas)

        self.lambda_L1 = lambda_L1

        self.loss_D_fake = torch.tensor(0.)
        self.loss_D_real = torch.tensor(0.)
        self.loss_D = torch.tensor(0.)

        self.loss_G_GAN = torch.tensor(0.)
        self.loss_G_L1 = torch.tensor(0.)
        self.loss_G = torch.tensor(0.)

    def set_requires_grad(self, model, requires_grad):
        for parameter in model.parameters():
            parameter.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data["L"].to(self.device)
        self.ab = data["ab"].to(self.device)

    def forward(self):
        self.fake_color = self.G(self.L).to(self.device)

    def backward_D(self):
        fake_images = torch.cat([self.L, self.fake_color], 1)
        fake_preds = self.D(fake_images.detach())

        self.loss_D_fake = self.GANcriterion(fake_preds, False)

        real_images = torch.cat([self.L, self.ab], 1)
        real_preds = self.D(real_images)

        self.loss_D_real = self.GANcriterion(real_preds, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) / 2

        self.loss_D.backward()

    def backward_G(self):
        fake_images = torch.cat([self.L, self.fake_color], 1)
        fake_preds = self.D(fake_images)

        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.lambda_L1 * self.L1criterion(self.fake_color, self.ab)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize(self):
        self.forward()

        self.D.train()
        self.set_requires_grad(self.D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.G.train()
        self.set_requires_grad(self.D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

def test():
    from discriminator import PatchDiscriminator
    from generator import pretrained_generator
    from torchvision.models.resnet import resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = PatchDiscriminator(1, 2)
    g = pretrained_generator(256, 1, 2, resnet18, -2)

    print(device)

    m = MainModel(device, d, g)
    data = {"L": torch.rand(2, 1, 256, 256), "ab": torch.rand(2, 2, 256, 256)}

    m.G.eval()

    with torch.no_grad():
        m.setup_input(data)
        m()

    m.G.train()

    print(m)
    print(m.fake_color.shape)

if __name__ == "__main__":
    test()
