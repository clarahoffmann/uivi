"""Utilities for loading MNIST data."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_mnist_dataloaders(img_size: int = 14, batch_size: int = 128):
    """Create MNIST dataloaders with flattened MNIST images of dimension
    img_size*img_size."""
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
