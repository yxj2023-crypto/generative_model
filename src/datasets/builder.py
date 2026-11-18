"""Torchvision datasets with native resolution and [0, 1] pixel values."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


DATASETS = {
    'mnist': datasets.MNIST,
    'fashion_mnist': datasets.FashionMNIST,
    'cifar10': datasets.CIFAR10,
    'celeba':datasets.CelebA
}


def build_dataset(name, root, train=True, download=True):
    dataset = DATASETS[name](root=str(root), train=train, download=download,
                            transform=transforms.ToTensor())
    num_classes = len(dataset.classes)
    return dataset, num_classes


def build_dataloader(name, root, batch_size=16, train=True, num_workers=0,
                     limit=None, download=True):
    dataset, num_classes = build_dataset(name, root, train=train, download=download)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                        num_workers=num_workers)
    return loader, num_classes
