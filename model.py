from utils import *

class TitanicModel(nn.Module):
    def __init__(self, input_size,**kwargs):
        super(TitanicModel, self).__init__()
        print(input_size)
        l = kwargs.get('layers')
        dropout = kwargs.get('dropout')
        self.layers = nn.Sequential(
                nn.Linear(input_size, l),
                nn.BatchNorm1d(l),  #Batchnorm best when applied before activation function
                nn.LeakyReLU(),
                nn.Dropout(p=0.2) if dropout else nn.Identity(),  #use dropout after activation  CONV / Dense -> BN -> ReLU -> Dropout
                nn.Linear(l,l*2),
                nn.BatchNorm1d(l*2),
                nn.Flatten(),
                nn.Linear(l*2, l),  
                nn.Linear(l,1), # Output layer with 1 neuron for binary classification
        )
        

    def forward(self, x):
        return self.layers(x)
