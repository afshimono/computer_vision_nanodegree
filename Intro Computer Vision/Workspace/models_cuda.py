## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
#input size = 224x224x3
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #output is (246-5)/1+1 = (32,242,242)
        self.conv1 = nn.Conv2d(1, 32, 5).cuda()
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool layer
        # pool with kernel_size=4, stride=2
        #output size= (32,121,121)
        self.pool1 = nn.MaxPool2d(4, 2).cuda()
        
        # second conv layer: 32 inputs, 68 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (121-5)/1 +1 = 117
        # the output tensor will have dimensions: (68, 117, 117)
        self.conv2 = nn.Conv2d(32, 68, 5).cuda()
        
        # maxpool layer
        # pool with kernel_size=5, stride=5
        #output size= (68,24,24)
        self.pool2 = nn.MaxPool2d(5, 5).cuda()
        
        # 68 outputs * the 10*10 filtered/pooled map size (max pool is used twice!)
        self.fc1 = nn.Linear(68*23*23, 1088).cuda()
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4).cuda()
        
        # finally, create 136 output channels 
        self.fc2 = nn.Linear(1088, 136).cuda()

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
