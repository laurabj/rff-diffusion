# Imports
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_circles, make_moons, make_s_curve, make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

############################# 2D Toy Datasets ##################################

# This function returns the x,y coordinates of n points sampled from a shape
# with two interleaving half-circles
def moons_dataset(n=8000):
    # Get x,y coordinates for n points in moon shape
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    # Old transformation (remove before submission)
    # X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    # X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    # Translate to centre moons and scale up
    X[:, 0] = (X[:, 0] - 0.5) * 2 # Transform x coords
    X[:, 1] = (X[:, 1] - 0.25) * 3 # Transform y coords
    return X

# This function returns the x,y coordinates of n points sampled from a shape of
# a vertical line.
def line_dataset(n=8000):
    # Create random generator with given random seed
    rng = np.random.default_rng(42)
    # Get x coords sampled uniformly between -1 and 1
    x = rng.uniform(-1, 1, n)
    # Get y coords sampled uniformly between -4 and 4
    y = rng.uniform(-4, 4, n)
    # Stack x and y coords
    X = np.stack((x, y), axis=1)
    return X

# This function returns the x,y coordinates of n points sampled from a shape with
# two interleaving circles.
def circle_dataset(n=8000):
    # Get x, y coordinates for n points in shape of circles
    X, _ = make_circles(n_samples=n, factor=0.5, noise=0.03)
    # Old transformation (remove before submission)
    # X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    # X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    # Scale up coords
    X[:, 0] = (X[:, 0]) * 3 # Transform x coords
    X[:, 1] = (X[:, 1]) * 3 # Transform y coords
    return X

# This function returns the x,y coordinates of n points sampled from a dino
# shape
def dino_dataset(n=8000):
    # Get the dino dataset from its file
    df = pd.read_csv("DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]
    print("number of datapoints in dino: " +str(len(df)))
    # Create random generator with given random seed
    rng = np.random.default_rng(42)
    # Get n random integers between 0 and number of points in dino
    ix = rng.integers(0, len(df), n)
    # Get x and y coords at these indices
    x = df["x"].iloc[ix].tolist()
    y = df["y"].iloc[ix].tolist()
    # Add random Gaussian noise
    x = np.array(x) + rng.normal(loc=0.0, scale=0.15, size=len(x))
    y = np.array(y) + rng.normal(loc=0.0, scale=0.15, size=len(x))
    # Old transformation (remove before submission)
    # x = (x/54 - 1) * 4
    # y = (y/48 - 1) * 4
    # Translate dino to approximate centre and scale down coords
    x = x/12 - 5
    y = y/12 - 4
    # Stack x and y coords
    X = np.stack((x, y), axis=1)
    return X

# This function returns the x,y coordinates of n points sampled from a swiss roll
# shape.
def spiral_dataset(n=8000):
    # Get x,y,z coordinates for n points in a swiss roll shape
    X, _ =  make_swiss_roll(n_samples=n, noise=0.03)
    # Extract appropriate x and y coords
    X = X[:,[0,2]]
    # Old transformation (remove before submission)
    # X /= 5
    # Scale down coords
    X /= 4
    return X

# This function returns the x,y coordinates of n points sampled from an s shape.
def s_dataset(n=8000):
    # Get x,y,z coordinates for n points in an s shape
    X, _ = make_s_curve(n_samples=n, noise=0.03)
    # Old transformation (remove before submission)
    # X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    # X[:, 2] = (X[:, 2] + 0.3) * 2 - 1
    # Extract appropriate x and y coords
    X = X[:,[0,2]]
    # Scale up coords
    X[:, 0] = (X[:, 0]) * 2 # Transform x coords
    X[:, 1] = (X[:, 1]) * 2 # Transform y coords
    return X

# This function returns n samples from the dataset indicated by the "name"
# argument.
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

############################ 2D Image Datasets #################################

def get_fashion_mnist_data(image_size=28, c=0, n=1000, num_classes=10):
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])])
    train_dataset=FashionMNIST(root="./fashion_mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    # Create balanced dataset
    if c == -1:
        data_list = []
        num_samples = n // num_classes
        for i in range(num_classes):
        # for i in [2, 8]:
            data = [torch.flatten(x[0], start_dim=1) for x in train_dataset if x[1] == i]
            data = torch.cat(data, dim=0)
            data = data[:num_samples]
            data_list.append(data)
        data = torch.cat(data_list, dim=0)
    # Only retain images of chosen class, and flatten images
    else:
        data = [torch.flatten(x[0], start_dim=1) for x in train_dataset if x[1] == c]
        data = torch.cat(data, dim=0)
        data = data[:n]

    return data


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
        for i in [2, 8]:
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
