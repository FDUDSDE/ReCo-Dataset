import torch
import os
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
import argparse
from model import Generator

ckpt_dir = os.path.join('checkpoints')
log_dir = os.path.join('logs')
generator_ckpt = 'generator.pth'
pic_name = 'result.png'

z_dim = 100
img_size = 128

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=img_size, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False


generator = Generator(opt)
generator.load_state_dict(torch.load(os.path.join(ckpt_dir, generator_ckpt), map_location=torch.device('cpu')))
generator.eval()


# Generate 1000 images and make a grid to save them.
n_output = 1000
z_sample = Variable(torch.randn(n_output, z_dim))
imgs_sample = (generator(z_sample).data + 1) / 2.0
filename = os.path.join(log_dir, pic_name)
torchvision.utils.save_image(imgs_sample, filename, nrow=10)

# Show 32 of the images.
grid_img = torchvision.utils.make_grid(imgs_sample[:32].cpu(), nrow=10)
plt.figure(figsize=(10, 10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()

