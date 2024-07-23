from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import torch

# Custom augmentation for per pixel mean subtract (couldn't find one in PyTorch, only one for per channel mean)
class SubtractMean(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, img):
        return img-self.mean
    
class MapDataset(torch.utils.data.Dataset):
    """
    Applies a transformation to an existing dataset

    (I'm really surprised by this, but PyTorch actually
    seems to have no way of applying a different transform
    to train and val data, without using a class like this...)
    """

    def __init__(self, dataset, trans):
        self.dataset = dataset
        self.trans = trans

    def __getitem__(self, index):
        item, label, = self.dataset[index]
        return (self.trans(item), label)

    def __len__(self):
        return len(self.dataset)

def calc_mean(dataset):
    mean = torch.zeros(3,32,32)
    for x in dataset:
        mean += x[0]
    mean /= len(dataset)
    torch.save(mean, "mean.pt")
    return mean

def loaders():
    try:
        mean = torch.load("mean.pt")
    except FileNotFoundError:
        dataset = CIFAR10("data", train=True, transform=transforms.ToTensor())
        mean = calc_mean(dataset)

    # Transformations used in the paper
    train_trans = transforms.Compose([
        transforms.RandomCrop(size=32,padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    base_trans = transforms.Compose([
        transforms.ToTensor(),
        SubtractMean(mean)
    ])

    # Import and split data, apply transforms
    (training_data, validation_data) = random_split(CIFAR10("data", train=True, transform=base_trans), [45000, 5000])
    test_data = CIFAR10("data", train=False, transform=base_trans)
    training_data = MapDataset(training_data, train_trans)

    # Create loaders
    training_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

    return (training_dataloader, validation_dataloader, test_dataloader)

"""(data , _, _) = loaders()

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
images, label = next(iter(data))
for i in range(1, cols * rows + 1):
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(images[i].permute(1,2,0), cmap="gray")
plt.show()"""