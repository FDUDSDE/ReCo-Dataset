from torch.utils.data import Dataset
import os
import glob
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision


class CommunityDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        # 1. Load the image
        img = torchvision.io.read_image(fname)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def get_dataset(root, img_size, gray=False):
    fnames = glob.glob(os.path.join(root, '*'))
    # 1. Resize the image to (64, 64)
    # 2. Linearly map [0, 1] to [-1, 1]
    compose = [
        transforms.ToPILImage(),
        transforms.CenterCrop(320),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    if gray:
        compose = [
            transforms.ToPILImage(),
            transforms.CenterCrop(320),
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    transform = transforms.Compose(compose)
    dataset = CommunityDataset(fnames, transform)
    return dataset


def show_some_pics(dataset):
    images = [dataset[i] for i in range(25)]
    grid_img = torchvision.utils.make_grid(images, nrow=5)
    plt.figure(figsize=(10, 10))

    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

