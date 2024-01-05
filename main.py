import click
import torch
from torch import nn, optim
from data import *
import matplotlib.pyplot as plt
from model import *

# Moving data to GPU if possible for higher computational speed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.group()
def cli():
    """Command line interface."""
    pass

"""
    This is a decorator provided by the click library, 
    which is used to define an option for a command-line interface. 
    It allows you to specify command-line arguments that can be passed to your script.
"""

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=64, help="batch size for training")
@click.option("--epochs", default=10, help="number of training epochs")
def train(lr, batch_size, epochs):
    # RUN CODE: python main.py train --lr 0.001 --batch_size 64 --epochs 10
    print('Training')
    print('learning rate', lr)
    print('batch size', batch_size)
    print('Epochs', epochs)

    # Able dropout during training 
    model = NN_Model()
    
    """
        Function mnist() returns two values, 
        and you're interested in capturing only the first of these returned values into the variable train_set. 
        The underscore _ is used as a placeholder for the second value which you're choosing to ignore.
    """
        
    """
        Dataset object train_set containing training data, which have been created using 'TensorDataset' 
        The DataLoader object is assigned to the variable trainloader. This object is iterable. 
        When you iterate over trainloader in a training loop, it will yield batches of data from train_set, 
        each batch being of size batch_size.
        
        You would iterate over trainloader to get your data in batches. For each iteration, 
        the trainloader provides a batch of data that you can feed into your model for training.
        This way of loading data is very efficient and helps in managing memory usage, 
        as only a portion of the dataset is loaded and processed at a time.
    """
    train_set, _ = mnist()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    
    # Type of loss function
    criterion = nn.NLLLoss()
    # Optimization algorithm used for training neural networks, by updating the weights and biases
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epochs):
        model.train()
        # The passing each time one batch of the trainloader, part of the whole data
        for batch in trainloader:
            optimizer.zero_grad()
            batch_data, batch_labels = batch
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {e} Loss {loss}")
        
    torch.save(model, "model.pt")

"""
    Model_checkpoint is used to pass the saved model you want to evaluate
"""    
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    # RUN CODE: python main.py evaluate {name of torch.save file}
    print(model_checkpoint)

    # Loading the model.pt file that is made and save in the train function
    model = torch.load(model_checkpoint)
    # Taking only the second return of mnist() function ignoring the first one
    _, test_set = mnist()
    # Creating dataloader called testloader for test_data
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    
    # Disable dropout during evaluation
    model.eval()
    
    # Lists to save results
    test_preds = [ ]
    test_labels = [ ]
     
    # Disabling autograd
    with torch.no_grad():
        # Looping through each batch of the testloader
        for batch in testloader:
            batch_data, batch_labels = batch
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_data)
            test_preds.append(logits.argmax(dim=1).cpu())
            test_labels.append(batch_labels.cpu())

    # Stack data vertically
    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    print((test_preds == test_labels).float().mean())


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()