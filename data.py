import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

def mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [ ], [ ]
    for i in range(5):
        # Appending the seperate data blocks into one list
        train_data.append(torch.load(f"train_images_{i}.pt"))
        train_labels.append(torch.load(f"train_target_{i}.pt"))

    # Stack data vertically 
    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    
    # train_data.shape is [25000, 28, 28], meaning 25000 samples, with each 28x28
    
    # Size of test_data is [5000, 28, 28]
    test_data = torch.load("test_images.pt") 
    test_labels = torch.load("test_target.pt")

    """
        -Add 1 dimension size become for train_data [25000, 1, 28, 28]
        -inserting a new dimension at the position 1 (position indexing starts at 0)
        - Is done to fit the shape of the model    
        - In the context of image data, this is often used to represent the number of channels in each image. 
        For grayscale images, there is only one channel, so this dimension is 1.
    """
    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    """
        A TensorDataset is a dataset wrapper around tensors. 
        Each element of a TensorDataset is a tuple containing one data point and its corresponding label. 
        It's a convenient way to pair your input data with labels.
        You are essentially creating an object that stores the train_data and train_labels tensors in a structured way. 
    """
    return (
        torch.utils.data.TensorDataset(train_data, train_labels), 
        torch.utils.data.TensorDataset(test_data, test_labels)
    )
