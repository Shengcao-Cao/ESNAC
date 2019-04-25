import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class CIFAR10(object):

    def __init__(self, batch_size=128, num_workers=4):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./datasets/data',
            train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./datasets/data',
            train=False, download=True, transform=test_transform)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True)

class CIFAR10Val(object):

    def __init__(self, batch_size=128, num_workers=4, val_size=5000):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./datasets/data',
            train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(root='./datasets/data',
            train=True, download=True, transform=val_transform)

        total_size = len(train_dataset)
        indices = list(range(total_size))
        train_size = total_size - val_size
        train_sampler = SubsetRandomSampler(indices[:train_size])
        val_sampler = SubsetRandomSampler(indices[train_size:])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
            sampler=val_sampler, num_workers=num_workers, pin_memory=True)