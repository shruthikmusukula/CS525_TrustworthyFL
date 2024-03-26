
import torch.nn.functional as F
import torch.nn as nn


class Model(nn.Module):
    '''
    Created a Client Model which goes through 3 layers
    - First layer is a linear layer based on the dimension of the mnist images (28x28)
    - Second layer takes the output of the first in order to break down dimensions
    - Final layer looks to output a number bw 0-9 i.e. 10 outputs
    
    '''
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  
        self.fc2 = nn.Linear(128, 64)   
        self.fc3 = nn.Linear(64, 10)      

    def forward(self, x):
        x = x.view(-1, 28*28)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return F.log_softmax(x, dim=1)  # Return log-probability
    

