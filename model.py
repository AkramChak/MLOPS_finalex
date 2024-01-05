from torch import nn
import torch.nn.functional as F

class NN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer
        self.fc1 = nn.Linear(784, 256)
        
        # Hidden layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        
        # Output layer, 10 outputs since 10 classes, each output represents the probability belonging to that class
        self.fc4 = nn.Linear(64, 10)
        
        """
            The process of dropping out neurons that specifies the probability of an input unit being dropped. 
            For example, a dropout rate of 0.2 means there's a 20% chance that any given unit is dropped during training.
            Dropout is particularly useful in large neural networks and is a key technique to combat overfitting, 
            ensuring that the model generalizes well to unseen data.
        """
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # flattening the tensor, to meet the required form of the nn model
        x = x.view(x.shape[0], -1)
        
        # nn network pattern
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x