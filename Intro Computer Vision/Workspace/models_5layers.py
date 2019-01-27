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
        #output is (246-5)/1+1 = (96,242,242)
        self.conv1 = nn.Conv2d(1, 96, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # first maxpool layer
        # pool with kernel_size=4, stride=4
        #output size= (96,61,61)
        self.pool1 = nn.MaxPool2d(4, 4)
        
        # second conv layer: 96 inputs, 192 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (61-5)/1 +1 = 57
        # the output tensor will have dimensions: (192, 57, 57)
        self.conv2 = nn.Conv2d(96, 192, 5)
        
        # second maxpool layer
        # pool with kernel_size=2, stride=2
        #output size= (192,29,29)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        
        
        # third conv layer: 48 inputs, 68 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (29-5)/1 +1 = 25
        # the output tensor will have dimensions: (192, 25, 25)
        self.conv3 = nn.Conv2d(192, 192, 5)
        
        # third maxpool layer
        # pool with kernel_size=2, stride=2
        #output size= (192,13,13)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # fourth conv layer: 68 inputs, 84 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (288, 11, 11)
        self.conv4 = nn.Conv2d(192, 288, 3)      
        
        
        # 68 outputs * the 8*8 filtered/pooled map size (max pool is used twice!)
        self.fc1 = nn.Linear(288*10*10, 1088)
        
        # dropout with p=0.1
        self.fc1_drop = nn.Dropout(p=0.5)
        
       
        self.fc2 = nn.Linear(1088, 136)
        
       

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two conv/relu + pool layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        #x = self.fc1_drop_cnn(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #x = self.fc1_drop_cnn(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        #x = self.fc1_drop_cnn(x)
        x = F.relu(self.conv4(x))


        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # four linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        

        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
