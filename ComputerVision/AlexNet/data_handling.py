import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def data_loaders(data_dir, batch_size):

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader
