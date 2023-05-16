

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_circles, make_moons, make_s_curve, make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST


def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return X
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return X
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    '''rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3'''
    X, _ = make_circles(n_samples=n, factor=0.5, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1

    return X
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000):
    df = pd.read_csv("DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return X
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def spiral_dataset(n=8000):
    X2, _ =  make_swiss_roll(n_samples=n, noise=0.03)
    X2 = X2[:,[0,2]]
    X2 /= 5
    return X2

def s_dataset(n=8000):
    X, _ = make_s_curve(n_samples=n, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1

    return X[:,0:2]

def get_dataset(name, n=8000):
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    elif name == "spiral":
        return spiral_dataset(n)
    elif name == "s":
        return s_dataset(n)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_mnist_data(image_size=28, digit=1, n=1000, num_digits=10):
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]
    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )

    # Create balanced dataset
    if digit == -1:
        data_list = []
        num_samples = n // num_digits
        # for i in range(num_digits):
        for i in [6, 7]:
            data = [torch.flatten(x[0], start_dim=1) for x in train_dataset if x[1] == i]
            data = torch.cat(data, dim=0)
            data = data[:num_samples]
            data_list.append(data)
        data = torch.cat(data_list, dim=0)
    # Only retain images of chosen digit, and flatten images
    else:
        data = [torch.flatten(x[0], start_dim=1) for x in train_dataset if x[1] == digit]
        data = torch.cat(data, dim=0)
        data = data[:n]

    return data

def get_mnist_dataloaders(batch_size,image_size=28,num_workers=4):

    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )
    idx = train_dataset.targets==1
    train_dataset.targets = train_dataset.targets[idx]
    train_dataset.data = train_dataset.data[idx]

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
