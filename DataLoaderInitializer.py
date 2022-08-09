import torch
import torchvision.datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms

from DatasetEnum import DatasetEnum
from SineWave import SineWave


class DataLoaderInitializer:

    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        self.train_dataset = None
        self.test_dataset = None
        self.trainloader = None
        self.testloader = None

        self.initialize_dataloader()

    def initialize_sine_wave_dataloader(self):
        _dataset = SineWave()

        # 0.8 means 80% train 20% test
        train_test_split_ratio = 0.8
        self.train_dataset, self.test_dataset = random_split(_dataset,
                                                   [round(len(_dataset) * train_test_split_ratio),
                                                    round(len(_dataset) * (1 - train_test_split_ratio))])

        self.trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def initialize_cifar_dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Lambda(lambda x: torch.flatten(x))])

        if self.dataset_name == "CIFAR10":
            self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                              download=True, transform=transform)
            self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                             download=True, transform=transform)

        else:
            self.train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                               download=True, transform=transform)
            self.test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                              download=True, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=0)

        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=0)

    def initialize_mnist_dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,)),
             transforms.Lambda(lambda x: torch.flatten(x))])

        self.train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def initialize_imagenet_dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]),
             transforms.Lambda(lambda x: torch.flatten(x))])

        self.train_dataset = torchvision.datasets.ImageNet('./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.test_dataset = torchvision.datasets.ImageNet('./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def initialize_dataloader(self):
        if self.dataset_name == "SINE":
            self.initialize_sine_wave_dataloader()

        if self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
            self.initialize_cifar_dataloader()

        if self.dataset_name == "MNIST":
            self.initialize_mnist_dataloader()

        if self.dataset_name == "IMAGENET":
            self.initialize_imagenet_dataloader()

    def get_datasets_and_dataloaders(self):
        return self.train_dataset, self.test_dataset, self.trainloader, self.testloader
