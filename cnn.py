import torch as torch
from process_data import *

class CNN(torch.nn.Module):
    """
    The Le-Net arquitecture consist in the following:

    INPUT ==> 5X5 CONV2D X6 ==> 2X2 MAXPOOL STRIDE 2 ==> 5X5 CONV2D X16 ==> 2X2 MAXPOOL STRIDE 2 ==>
    ==> FLATTEN ==> FC 5*5*16X120 ==> FC 120X64 ==> FC 64XCLASSES

    As the torch's CrossEntropy loss takes as inputs the logits, don't apply SoftMax. After each 
    convolution, a ReLU activation is applyed before the pooling. 

    The operation 5*5*16 comes as the result of applying each convolution and pooling, leading to 
    the total transformation: 32X32X1 ==> 5X5X16.
    """

    def __init__(self, in_channels, classes):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.conv1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = 6, kernel_size = (5, 5), stride = (1,1))
        self.pool1 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)) # Default value of strides is kernel_size
        self.conv2 = torch.nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), stride = (1,1))
        self.pool2 = torch.nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.flatten = torch.nn.Flatten()
        self.FC3 = torch.nn.Linear(in_features = 5*5*16, out_features = 120)
        self.FC4 = torch.nn.Linear(in_features = 120, out_features = 64)
        self.FC5 = torch.nn.Linear(in_features = 64, out_features = classes)
        
    def forward(self, x):
        # First convolutional layer
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool1(x)

        # Second convolutional layer
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool2(x)

        # Fully connected layers
        x = torch.flatten(x, 1) # flatten all except batch
        x = self.FC3(x)
        x = self.FC4(x)
        logits = self.FC5(x)
        return logits


