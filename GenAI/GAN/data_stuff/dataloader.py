import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def data_loader(batch_size, dataset_path):
    transform = transforms.Compose([ transforms.Resize(28),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    dataset = MNIST(root=dataset_path, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
