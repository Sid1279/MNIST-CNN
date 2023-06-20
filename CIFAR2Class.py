from torch.utils.data import Dataset


# Encapsulate data in a custom class to be used with PyTorch's DataLoader
class CIFAR2ClassDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        return image, label
