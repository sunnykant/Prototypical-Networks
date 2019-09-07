#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# ## ProtoNet Paper Model

# In[3]:


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)    
    )

class ProtoNet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=16, c_dim=1024, o_dim=256):
        super(ProtoNet,self).__init__()
        self.common_dim = c_dim
        self.output_dim = o_dim
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )
        
        self.dense_flowers = nn.Sequential(
            nn.Linear(c_dim, o_dim),
            nn.ReLU(),
        )
        
        self.dense_birds = nn.Sequential(
            nn.Linear(c_dim, o_dim),
            nn.ReLU(),
        )
        
        self.dense_dogs = nn.Sequential(
            nn.Linear(c_dim, o_dim),
            nn.ReLU(), 
        )
        
        self.dense_cars = nn.Sequential(
            nn.Linear(c_dim, o_dim),
            nn.ReLU(),
        )
        
        self.dense_aircrafts = nn.Sequential(
            nn.Linear(c_dim, 512),
            nn.ReLU(),
            nn.Linear(512, o_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        z_common = x.view(x.size(0), -1)
        z_flowers = self.dense_flowers(z_common)
        z_birds = self.dense_birds(z_common)        
        z_dogs = self.dense_dogs(z_common)        
        z_cars = self.dense_cars(z_common)        
        z_planes = self.dense_aircrafts(z_common)
        
        return z_common, z_flowers, z_birds, z_dogs, z_cars, z_planes 
#         return z_birds


# ## ProtoResNet - I

# In[111]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class ProtoImageNet(nn.Module):
    def __init__(self, model_name, img_dims=[3,318,318], o_dim=64, c_dim=256, feature_extract=True, is_pretrained=True):
        super(ProtoImageNet,self).__init__()
        self.model_name = model_name
        
        self.model = None
        
        if(model_name == 'resnet'):
            self.model = models.resnet34(pretrained=is_pretrained)
            set_parameter_requires_grad(self.model, feature_extract)
            num_features = self.model.fc.in_features
#             self.model.fc = Identity()
            self.model.fc = nn.Linear(num_features, c_dim)
        
        elif(model_name == 'vgg'):
            pass
        
        
        self.common_dim = self.model(torch.randn(1,img_dims[0],img_dims[1],img_dims[2])).reshape(-1).size()[0]
        self.output_dim = o_dim
        
        self.dense_flowers = nn.Sequential(
            nn.Linear(self.common_dim, o_dim),
            nn.ReLU(),
        )
        
        self.dense_birds = nn.Sequential(
            nn.Linear(self.common_dim, o_dim),
            nn.ReLU(),
        )
        
        self.dense_dogs = nn.Sequential(
            nn.Linear(self.common_dim, o_dim),
            nn.ReLU(), 
        )
        
        self.dense_cars = nn.Sequential(
            nn.Linear(self.common_dim, o_dim),
            nn.ReLU(),
        )
        
        self.dense_aircrafts = nn.Sequential(
            nn.Linear(self.common_dim, o_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        z_common = self.model(x)
        z_flowers = self.dense_flowers(z_common)
        z_birds = self.dense_birds(z_common)        
        z_dogs = self.dense_dogs(z_common)        
        z_cars = self.dense_cars(z_common)        
        z_planes = self.dense_aircrafts(z_common)
        
        return z_common, z_flowers, z_birds, z_dogs, z_cars, z_planes 
    


# In[93]:


net = ProtoImageNet(model_name='resnet')
print(net.model)


# In[102]:


x = torch.randn((1,3,224,224))
net.model(x)[0].size()


# In[120]:


mm = ProtoImageNet('resnet')
print(mm.model)


# In[119]:


new_model = nn.Sequential(*list(mm.model.children())[:-1])
print(new_model)


# In[61]:


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)    
    )

class ProtoNet(nn.Module):
    """
        img_dims = [img_channels, img_height, img_width]      
    """
    def __init__(self, img_dims, hid_channels=64, out_channels=16, o_dim=256):
        super(ProtoNet,self).__init__()
        
        self.in_channels = img_dims[0]
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        
        self.encoder = nn.Sequential(
            conv_block(self.in_channels, self.hid_channels),
            conv_block(self.hid_channels, self.hid_channels),
            conv_block(self.hid_channels, self.hid_channels),
            conv_block(self.hid_channels, self.hid_channels),
            conv_block(self.hid_channels, self.out_channels)
        )
        
        self.output_dim = o_dim
        self.common_dim = self.encoder(torch.randn(1,img_dims[0],img_dims[1],img_dims[2])).reshape(-1).size()[0]
        
        self.dense_flowers = nn.Sequential(
            nn.Linear(self.common_dim, self.output_dim),
            nn.ReLU(),
        )
        
        self.dense_birds = nn.Sequential(
            nn.Linear(self.common_dim, self.output_dim),
            nn.ReLU(),
        )
        
        self.dense_dogs = nn.Sequential(
            nn.Linear(self.common_dim, self.output_dim),
            nn.ReLU(), 
        )
        
        self.dense_cars = nn.Sequential(
            nn.Linear(self.common_dim, self.output_dim),
            nn.ReLU(),
        )
        
        self.dense_aircrafts = nn.Sequential(
            nn.Linear(self.common_dim, self.output_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        z_common = x.view(x.size(0), -1)
        z_flowers = self.dense_flowers(z_common)
        z_birds = self.dense_birds(z_common)        
        z_dogs = self.dense_dogs(z_common)        
        z_cars = self.dense_cars(z_common)        
        z_planes = self.dense_aircrafts(z_common)
        
        return z_common, z_flowers, z_birds, z_dogs, z_cars, z_planes 
#         return z_birds


# In[60]:


i_dims = [3, 224, 224]
m = ProtoNet(img_dims=i_dims)
x = torch.randn((1,i_dims[0], i_dims[1], i_dims[2]))
y = m(x)


# In[ ]:




