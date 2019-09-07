#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

import time
import os.path as osp
from PIL import Image
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# In[ ]:


class_data_path = '/data/skbharti/Fine-ProtoNet/data'
MODEL_PATH = 'models/proto_model.pt'
PROTOTYPES_PATH = 'models/prototypes.pt'


rt={'birds'    : osp.join(class_data_path,'class_data/CUB_200_2011'),
    'dogs'     : osp.join(class_data_path,'class_data/dogs'),
    'aircrafts': osp.join(class_data_path,'class_data/aircrafts'),
    'cars'     : osp.join(class_data_path,'class_data/cars')
   }
id_class_dict = {}
        
file_path1 = osp.join(rt['birds'],'classes.txt')
lines1 = [(line.strip()).split(" ",1) for line in open(file_path1, 'r').readlines()]
        
file_path2 = osp.join(rt['dogs'],'classes.txt')
lines2 = [(line.strip()).split(" ",1) for line in open(file_path2, 'r').readlines()]
        
file_path3 = osp.join(rt['aircrafts'],'classes.txt')
lines3 = [(line.strip()).split(" ",1) for line in open(file_path3, 'r').readlines()]
        
file_path4 = osp.join(rt['cars'],'classes.txt')
lines4 = [(line.strip()).split(" ",1) for line in open(file_path4, 'r').readlines()]
        
mat = lines1 + lines2 +lines3 + lines4
total_subclasses = len(mat)

new_mat = [] 

lines1 = None
del lines1
lines2 = None
del lines2
lines3 = None
del lines3
lines4 = None
del lines4

max1=0
max2=0
max3=0
max4=0

def coarseclass_name(x): #x is id>=1
    if (x<=max1):
        return "Birds"
    elif (x<=max2):
        return "Dogs"
    elif (x<=max3):
        return "Aircrafts"
    elif (x<=max4):
        return "Cars"
    
def subclass_name(x): # x is id >=1
    class_name = new_mat[x-1]
    return class_name


# In[3]:


img_height, img_width, img_channels = 318, 318, 3

class Dataset2(Dataset):
    
    def __init__(self,data_type,classes):  
        
        
        global new_mat
        new_mat = []
        data_leng = []
        #all_data = [[ ]]
        for i in range(len(classes)):
            file_path = osp.join(rt[classes[i]],'img_label_train_test.txt')
            line = [(line.strip()).split() for line in open(file_path, 'r').readlines()]
            if(i!=0):
                all_data = all_data + line
            else:
                all_data = line
            data_leng.append(len(all_data))
    
        
        tmp_tdata =[]
        tmp_tlabel =[]
        tmp_tstdata =[]
        tmp_tstlabel =[]
        tmp_vdata =[]
        tmp_vlabel =[]
        
        j=0
        for i in range(len(all_data)):
            for k in range(len(data_leng)):
                if i< data_leng[k]:
                    root = rt[classes[k]]
                    break
                    
            if int(all_data[i][2])==1:
                
                path =  osp.join(root,'images',str(all_data[i][0]))
                j = j+1
                if(j%4!= 0):
                    tmp_tdata.append(path)
                    tmp_tlabel.append(int(all_data[i][1]))
                else:
                    tmp_vdata.append(path)
                    tmp_vlabel.append(int(all_data[i][1]))
                    
                    
            elif int(all_data[i][2])==0:
                path =  osp.join(root,'images',str(all_data[i][0]))
                tmp_tstdata.append(path)
                tmp_tstlabel.append(int(all_data[i][1]))
               
                
                
        k_min = 20       
        lb=1
        unq = np.unique(tmp_tlabel)
        for i in unq:  
            ind = np.argwhere(np.array(tmp_tlabel) == i).reshape(-1)   
            ind.sort()
            if (len(ind)<k_min):
                
                p=0
                for j in ind:
                    del tmp_tlabel[j-p]
                    del tmp_tdata[j-p]
                    p=p+1
                    
                tstind = np.argwhere(np.array(tmp_tstlabel) == i).reshape(-1) 
                tstind.sort()
                p=0
                for j in tstind:
                    del tmp_tstlabel[j-p]
                    del tmp_tstdata[j-p]
                    p=p+1
                    
                vind = np.argwhere(np.array(tmp_vlabel) == i).reshape(-1) 
                vind.sort()
                p=0
                for j in vind:
                    del tmp_vlabel[j-p]
                    del tmp_vdata[j-p]
                    p=p+1
                    
            else:
                global mat
                new_mat.append(mat[i-1][1])
                if(i!=lb):
                    
                    for k in ind:
                        tmp_tlabel[k]=lb
                    
                    tstind = np.argwhere(np.array(tmp_tstlabel) == i).reshape(-1) 
                    tstind.sort()
                    for k in tstind:
                        tmp_tstlabel[k]=lb
                
                    vind = np.argwhere(np.array(tmp_vlabel) == i).reshape(-1) 
                    vind.sort()
                    for k in vind:
                        tmp_vlabel[k] = lb
                
                if i<=200:  # make sure order birds, dogs, aircrafts, cars
                    global max1
                    max1= lb
                    
                elif i<=320:
                    global max2
                    max2= lb
                    
                elif i<=390:
                    global max3
                    max3= lb
                        
                elif i<=586:
                    global max4
                    max4= lb
                    
                lb = lb+1
                
                
        l1,l2,l3,l4=0,0,0,0 
        uq,count = np.unique(tmp_tlabel, return_counts = True)
        for i,u in enumerate(uq):
            if(u<=max1):
                l1= l1+ count[i]
            elif(u<=max2):
                l2= l2+ count[i]
            elif(u<=max3):
                l3= l3+ count[i]
            elif(u<=max4):
                l4= l4+ count[i]
                
                
        #print(max1,max2,max3,max4)   
        #print(len(np.unique(tmp_tlabel)))
        
        leng =[]     
        if(l1!=0):
            leng.append(l1)
        if(l2!=0):
            leng.append(l2)
        if(l3!=0):
            leng.append(l3)
        if(l4!=0):
            leng.append(l4)
        for i in range(len(leng)-1):
            leng[i+1]= leng[i+1] + leng[i]
            
        self.leng = leng
        
        if data_type == "train":
            self.data = tmp_tdata 
            self.label = tmp_tlabel        
            self.length = len(np.unique(tmp_tlabel))
            
        elif data_type == "val":
            self.data = tmp_vdata 
            self.label = tmp_vlabel
            self.length = len(np.unique(tmp_vlabel))
            
        if data_type == "test":
            self.data = tmp_tstdata 
            self.label = tmp_tstlabel
            self.length = len(np.unique(tmp_tstlabel))
        
        #random horizontal flip can be added
        self.transform = transforms.Compose([
            transforms.Resize((318,318)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))  
        return image, label


# In[4]:


import torch
import numpy as np


class Batch_Sampler():

    def __init__(self, label, leng, n_cls, n_per,n_batch):
        self.n_batch = n_batch
        self.n_cls = n_cls   
        self.n_per = n_per
        self.leng = leng

        label = np.array(label)
        self.m_ind = {}  
        length=[]
        length.append(0)
        for i in leng:
            length.append(i)
        #print("batch",length)
    
        labels = {}
        for i in range(len(leng)):
            labels[str(i+1)] = label[length[i]:length[i+1]]
        
        self.m_ind={}
        
        for j in range(len(leng)):
            indx=[]
            for i in np.unique(labels[str(j+1)]):  
                ind = np.argwhere(label == i).reshape(-1)   
                ind = torch.from_numpy(ind)
                indx.append(ind)
            self.m_ind[str(j+1)] = indx
        
        #print(len(self.m_ind['1']),len(self.m_ind['2']))
        mnn=50000  #just to find minimum number of data points in subclass of all classes
        for i in range(len(leng)):
            s  =str(i+1)
            arr = self.m_ind[s]
            for l in range(len(arr)):
                mn = len(arr[l])
                if(mn<mnn):
                    mnn= mn
                    if mnn==0:
                        print(i+1," ",l," ",arr[l])
        print("maximum support + query = ",mnn)
        #print(np.unique(labels['1']),np.unique(labels['2']))
        
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch): 
            batch = [] 
            n= len(self.leng)
            rem = i_batch % n
            classes = torch.randperm(len(self.m_ind[str(rem +1)]))[:self.n_cls]  
            for c in classes:
                l = self.m_ind[str(rem +1)][c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            
class vt_Sampler():

    def __init__(self, label):
        
        label = np.array(label)
        indx=[]
        
        for i in (np.unique(label)): 
            ind = np.argwhere(label == i).reshape(-1)  
            ind = torch.from_numpy(ind)
            indx.append(ind)
            
        self.indx = indx
        self.length = len(indx)
        print(self.length)
    def __len__(self):
        return self.length
    
    def __iter__(self):
        for i_batch in range(self.length): 
            batch = []
            l = self.indx[i_batch]
            batch.append(l)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


# ## Define Models

# In[5]:


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


# In[6]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class ProtoImageNet(nn.Module):
    def __init__(self, model_name, o_dim=64, c_dim=256, feature_extract=True, is_pretrained=True):
        super(ProtoImageNet,self).__init__()
        self.model_name = model_name
        
        self.model = None
        self.common_dim = c_dim
        self.output_dim = o_dim
        
        if(model_name == 'resnet'):
            self.model = models.resnet34(pretrained=is_pretrained)
            set_parameter_requires_grad(self.model, feature_extract)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, c_dim)
        
        elif(model_name == 'vgg'):
            pass
        
        
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
            nn.Linear(c_dim, o_dim),
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
    


# ## Utility Functions

# In[7]:


def distance_metric(query_embeddings, prototypes):
#     print(query_embeddings.size(), prototypes.size())
    query_count = query_embeddings.size()[0]
    proto_count = prototypes.size()[0]
    
    a = query_embeddings.unsqueeze(1).expand(query_count, proto_count, -1)
    b = prototypes.unsqueeze(0).expand(query_count, proto_count, -1)
    
    logits = -((a - b)**2).sum(dim=2)
    return logits
    
    
def count_acc(logits, labels):
    print(logits.size(), labels.size())
    preds = torch.argmax(logits, dim=1)
    print(labels,'\n',preds)
    return (preds == labels).type(torch.cuda.FloatTensor).mean().item()

def count_subclass_acc(logits, labels, n_sub_classes):
    preds = torch.argmax(logits, dim=1) 
    result = torch.ones(n_sub_classes).to(device)
    
    samples_per_subclass = int(int(labels.size()[0])/n_sub_classes)
    for sc in range(n_sub_classes):
        sc_beg, sc_end = sc*samples_per_subclass, (sc+1)*samples_per_subclass 
        result[sc] = (preds[sc_beg:sc_end] == labels[sc_beg:sc_end]).type(torch.cuda.FloatTensor).mean().item()
    return result


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
class Maximizer():

    def __init__(self):
        self.v = 0

    def add(self, x):
        self.v = max(self.v, x)

    def item(self):
        return self.v


# In[8]:


debug_size = False
save_model = False
reload_model = True

MODEL_PATH = 'models/proto_model'+'_aircrafts_resnet34_epoch_1k_working'+'.pt'
PROTOTYPES_PATH = 'models/prototypes'+'_aircrafts_resnet34_epoch_1k_working'+'.pt'
OPTIMIZER_PATH = 'models/optimizer'+'_aircrafts_resnet34_epoch_1k_working'+'.pt'
ACCURACY_PATH = 'models/accuracy'+'_aircrafts_resnet34_epoch_1k_working'+'.pt'


# In[ ]:


device = torch.device("cuda:0")

n_sub_classes = 20
support = 30
query = 10
samples_per_class = support + query # support + query exapmles
episodes = 10
alpha = 0.5

# make sure order birds, dogs, aircrafts, cars

trainset = Dataset2('train',['aircrafts'])  # list can contain 'dogs','aircrafts','cars','birds'
train_sampler = Batch_Sampler(trainset.label,trainset.leng, n_sub_classes, samples_per_class, episodes)
total_training_subclass = len(np.unique(trainset.label))

valset = Dataset2('val',['aircrafts'])
val_sampler = vt_Sampler(valset.label)

# model = ProtoNet().cuda()
net = ProtoImageNet('resnet')
model = torch.nn.DataParallel(net)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

subclass_prototypes = torch.zeros((total_subclasses, net.output_dim)).to(device)
train_accuracy = {"Birds": Averager(), "Dogs": Averager(), "Cars": Averager(), "Aircrafts": Averager()}
val_accuracy = {"Birds": Averager(), "Dogs": Averager(), "Cars": Averager(), "Aircrafts": Averager()}

if(reload_model):
    model = torch.load(MODEL_PATH)
    optimizer = torch.load(OPTIMIZER_PATH)
    subclass_prototypes = torch.load(PROTOTYPES_PATH)
    print("Model Reloaded!")


# In[19]:


print(net)


# In[10]:


for n,p in model.named_parameters():
    if p.requires_grad:
         print(n)


# ## Training Model on Single class dataset

# In[13]:


num_epochs = 1000

accuracy_dic = {i+1:Averager() for i in range(total_training_subclass)}

for epoch in range(num_epochs):
#     print('===============epoch================ ',epoch)
#     lr_scheduler.step()
    
    is_sampled = torch.zeros(total_training_subclass).to(device)
    
    model.train()
    print("Training:")
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)
    
    # each batch is an episode consisting of one class with 'n_sub_classes' number of fine classes in that class.
    for i_batch, batch in enumerate(train_loader):
        start = time.time()
        print('=============epoch {0:2d} & batch {1:2d}============ '.format(epoch+1,i_batch+1))
        x, y = batch
        
        if(debug_size):
            print("\n\nBatch size from data : ",x.size(), y.size())
            print(y)
        
        coarse_class = coarseclass_name(y[0])
    
        x = x.to(device).reshape((samples_per_class, n_sub_classes, img_channels, img_height, img_width)).permute(1, 0, 2, 3, 4)
        y = y.to(device).reshape((samples_per_class, n_sub_classes)).permute(1, 0)
        # x/y is now n_sub_classes * samples_per_class * image/label
        
        if(debug_size):
            print("\n\nBatch size after 5-D stacking : ", x.size(), y.size())
            print(y)

        x_support = x[:,:support,:,:,:].reshape(n_sub_classes*support,img_channels, img_height, img_width)        
        x_query = x[:,support:,:,:,:].reshape(n_sub_classes*query,img_channels, img_height, img_width)
        y_support = y[:,:support].reshape(n_sub_classes*support)  
        y_query = y[:,support:].reshape(n_sub_classes*query)
        #x_support/y_support is now n_sub_classes * support * image/label
        
        if(debug_size):
            print("\n\nBatch support size of model input : ",x_support.size(), y_support.size())
            print(y_support)
            print("Batch query size of model input : ",x_query.size(), y_query.size())
            print(y_query)
        
        _, _, _, _, _, z_support  = model(x_support)
        _, _, _, _, _,   z_query  = model(x_query)
        
       
        
        z_support = z_support.reshape((n_sub_classes, support, -1))
        prototypes = torch.mean(z_support, dim=1)
        # prototypes dim is n_sub_classes * latent_len
        
        if(debug_size):
            print("\n\nSize of batch prototypes : ",prototypes.size())
        
        # update global subclass prototypes - subclasses are labels 1-n but prototype array is indexed 0-n-1
        for l in range(n_sub_classes):
            subclass_prototypes[y_support[l*support]-1] = alpha*subclass_prototypes[y_support[l*support]-1] + (1-alpha)*prototypes[l]   #subclass_prototypes are 0 indexed but y is 1 indexed
            
        
        logits = distance_metric(z_query, prototypes)
        labels = torch.Tensor([i for i in range(n_sub_classes)]).unsqueeze(1).expand(n_sub_classes, query).reshape(-1).type(torch.cuda.LongTensor)
        # creating labels tensor to calculate loss
        
        if(debug_size):
            print("\n\nBatch query logits size : ",logits.size())
            print("Batch query labels size & labels : ",labels.size())
            print(labels)
            
        loss = F.cross_entropy(logits, labels)
        
#         batch_all_accuracy = count_acc(logits, labels)
#         print(batch_all_accuracy)
        batch_subclass_accuracy = count_subclass_acc(logits, labels, n_sub_classes)
        print(batch_subclass_accuracy)
        
        # to estimate how many subclasses are sampled over episodes - for single class data only
        # update each subclass accuracy
        for sub_class in range(n_sub_classes):
            is_sampled[y[sub_class][0]-1] = 1
            accuracy_dic[int(y[sub_class][0])].add(batch_subclass_accuracy[sub_class])
            # update the sub_class accuracy in accuracy dictionary
            
#         print(acc)
        
    
        # update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prototypes = None; logits = None; loss = None
        
        # printing batch stats
        if((i_batch+1)%1==0):
            for key in accuracy_dic.keys():
                print("{0:2d} : {1:.2f}".format(key, accuracy_dic[key].item()))
            
#             print("Episode : {0:2d} - Loss : {1:.5f} - Accuracy : {2:.2f}".format(i_batch, loss, batch_accuracy))    
            print("{0:.0f} subclasses explored out of {1:2d} subclasses".format(torch.sum(is_sampled), total_training_subclass))
        
        print("Episode Time : ",time.time()-start)
        debug_size = False

if(save_model):
    # save model
    torch.save(model, MODEL_PATH)
    torch.save(subclass_prototypes, PROTOTYPES_PATH)
    torch.save(optimizer, OPTIMIZER_PATH)
    torch.save(accuracy_dic, ACCURACY_PATH)


# ## Best accuracy results obtained

# Individual Dataset
# -------------------------------------------------------
# Aircraft :  0.63 (using conv_blocks with resnet34)
# Birds    :  0.52 (using conv_blocks with resnet34)
# Dogs     :  0.35 (less trained conv_block only)
# Cars     :  0.31 (less trained conv_blocks only)
# 
# 
# All Dataset (using conv_blocks with resnet34)
# -------------------------------------------------------
# Aircrafts  - 0.25
# Birds      - 0.30 
# Dogs       - 0.27 
# Cars       - 0.21 

# In[ ]:


# def count_subclass_acc(logits, labels, n_sub_classes):
#     preds = torch.argmax(logits, dim=1) 
#     result = torch.ones(n_sub_classes).to(device)
    
#     samples_per_subclass = int(int(labels.size()[0])/n_sub_classes)
#     for sc in range(n_sub_classes):
#         sc_beg, sc_end = sc*samples_per_subclass, (sc+1)*samples_per_subclass 
#         result[sc] = (preds[sc_beg:sc_end] == labels[sc_beg:sc_end]).type(torch.cuda.FloatTensor).mean().item()
#     return result


# ## Extra code for individual dataset

# In[ ]:


# model.eval()
# print("Validation:")
# val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=8)
# val_accuracy_dic = {i+1:Averager() for i in range(total_training_subclass)}

# for j, batch in enumerate(val_loader):
#     x, y = batch
#     x = x.to(device)
#     y = y.to(device)

#     labels = y

#     _, _, _, _, _, z  = model(x)

#     logits = distance_metric(z, subclass_prototypes)

#     if(debug_size):
#         print("\n\nValidation Logits size and Labels size",logits.size(), labels.size())

#     loss = F.cross_entropy(logits, labels)
#     val_batch_accuracy = count_acc(logits, labels)
#     val_accuracy_dic[int(y[0])].add(val_batch_accuracy)
    
# for key in val_accuracy_dic.keys():
#     print("{0:2d} - {1:.2f}".format(key, val_accuracy_dic[key].item()))


# 
# ## Training Models on all class dataset

# In[ ]:


debug_size = False

num_epochs = 1000
total_training_subclass = len(np.unique(trainset.label))

accuracy_dic = {i+1:Averager() for i in range(total_training_subclass)}

for epoch in range(num_epochs):
#     print('===============epoch================ ',epoch)
#     lr_scheduler.step()
    
    is_sampled = torch.zeros(total_training_subclass).to(device)
    
    model.train()
    print("Training:")
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)
    
    # each batch is an episode consisting of one class with 'n_sub_classes' number of fine classes in that class.
    start = time.time()
    for i_batch, batch in enumerate(train_loader):
        print('=============epoch {0:2d} & batch {1:2d}============ '.format(epoch+1,i_batch+1))
        x, y = batch
        
        if(debug_size):
            print("\n\nBatch size from data : ",x.size(), y.size())
            print(y)
        
        coarse_class = coarseclass_name(y[0])
        
        if(not coarse_class=='Aircrafts'):
            continue
        
        
        x = x.to(device).reshape((samples_per_class, n_sub_classes, img_channels, img_height, img_width)).permute(1, 0, 2, 3, 4)
        y = y.to(device).reshape((samples_per_class, n_sub_classes)).permute(1, 0)
        # x/y is now n_sub_classes * samples_per_class * image/label
        
        if(debug_size):
            print("\n\nBatch size after 5-D stacking : ", x.size(), y.size())
            print(y)

        x_support = x[:,:support,:,:,:].reshape(n_sub_classes*support,img_channels, img_height, img_width)        
        x_query = x[:,support:,:,:,:].reshape(n_sub_classes*query,img_channels, img_height, img_width)
        y_support = y[:,:support].reshape(n_sub_classes*support)  
        y_query = y[:,support:].reshape(n_sub_classes*query)
        #x_support/y_support is now n_sub_classes * support * image/label
        
        if(debug_size):
            print("\n\nBatch support size of model input : ",x_support.size(), y_support.size())
            print(y_support)
            print("Batch query size of model input : ",x_query.size(), y_query.size())
            print(y_query)
        
        
        if(coarse_class=='Flowers'):
            _, z_support, _, _, _, _  = model(x_support)
            _,   z_query, _, _, _, _  = model(x_query)
            
        elif(coarse_class=='Birds'):
            _, _, z_support, _, _, _  = model(x_support)
            _, _,   z_query, _, _, _  = model(x_query)
        
        elif(coarse_class=='Dogs'):
            _, _, _, z_support, _, _  = model(x_support)
            _, _, _,   z_query, _, _  = model(x_query)
            
        elif(coarse_class=='Cars'):
            _, _, _, _, z_support, _  = model(x_support)
            _, _, _, _,   z_query, _  = model(x_query)
        
        elif(coarse_class=='Aircrafts'):
            _, _, _, _, _, z_support  = model(x_support)
            _, _, _, _, _,   z_query  = model(x_query)
        
       
        
        z_support = z_support.reshape((n_sub_classes, support, -1))
        prototypes = torch.mean(z_support, dim=1)
        # prototypes dim is n_sub_classes * latent_len
        
        if(debug_size):
            print("\n\nSize of batch prototypes : ",prototypes.size())
        
        # update global subclass prototypes - subclasses are labels 1-n but prototype array is indexed 0-n-1
        for l in range(n_sub_classes):
            subclass_prototypes[y_support[l*support]-1] = alpha*subclass_prototypes[y_support[l*support]-1] + (1-alpha)*prototypes[l]   #subclass_prototypes are 0 indexed but y is 1 indexed
            
        
        logits = distance_metric(z_query, prototypes)
        labels = torch.Tensor([i for i in range(n_sub_classes)]).unsqueeze(1).expand(n_sub_classes, query).reshape(-1).type(torch.cuda.LongTensor)
        # creating labels tensor to calculate loss
        
        if(debug_size):
            print("\n\nBatch query logits size : ",logits.size())
            print("Batch query labels size & labels : ",labels.size())
            print(labels)
            
        loss = F.cross_entropy(logits, labels)
        
        batch_accuracy = count_acc(logits, labels)
        
        
#         train_accuracy[coarse_class].add(batch_accurracy)   # required for multiple classes
        
        # to estimate how many subclasses are sampled over episodes - for single class data only
        for sub_class in range(n_sub_classes):
            is_sampled[y[sub_class][0]-1] = 1
            accuracy_dic[int(y[sub_class][0])].add(batch_accuracy)
            # update the sub_class accuracy in accuracy dictionary
            
#         print(acc)
        
    
        # update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prototypes = None; logits = None; loss = None
        
        # printing batch stats
        if((i_batch+1)%1==0):
            for key in accuracy_dic.keys():
                print("{0:2d} : {1:.2f}".format(key, accuracy_dic[key].item()))
            
#             print("Episode : {0:2d} - Loss : {1:.5f} - Accuracy : {2:.2f}".format(i_batch, loss, batch_accuracy))    
            print("{0:.0f} subclasses explored out of {1:2d} subclasses".format(torch.sum(is_sampled), total_training_subclass))

        debug_size = False



    model.eval()
    print("Validation:")
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=8)
    
    if(not i%10==0):
        continue
        
    for j, batch in enumerate(val_loader):
        x, y = batch
        coarse_class = coarseclass_name(y[0])
        
        if(not coarse_class=='Aircrafts'):
            continue
            
        x = x.to(device)
        y = y.to(device)
        
        labels = y
        
        if(coarse_class=='Flowers'):
            _, z, _, _, _, _  = model(x)
            
        elif(coarse_class=='Birds'):
            _, _, z, _, _, _  = model(x)
        
        elif(coarse_class=='Dogs'):
            _, _, _, z, _, _  = model(x)
            
        elif(coarse_class=='Cars'):
            _, _, _, _, z, _  = model(x)
        
        elif(coarse_class=='Aircrafts'):
            _, _, _, _, _, z  = model(x)
    
        logits = distance_metric(z, subclass_prototypes)
#         print(logits.size(), labels.size())
        loss = F.cross_entropy(logits, labels)
        acc = count_acc(logits, labels)
#         print(acc)
        val_accuracy[coarse_class].add(acc)
        
    print("Time taken : ",time.time()-start)
        print("{0:.3f} {1:.3f}".format(train_accuracy['Aircrafts'].item(), val_accuracy['Aircrafts'].item()))
    for i, key in enumerate(train_accuracy.keys()):
        print("{0:10s} - {1:.2f} ".format(key, train_accuracy[key].item()))
        

# save model
torch.save(model, MODEL_PATH)
torch.save(subclass_prototypes, PROTOTYPES_PATH)


# In[ ]:


# MODEL_PATH = 'models/proto_model'+'_aircrafts_resnet34_epoch_1k'+'.pt'
# PROTOTYPES_PATH = 'models/prototypes'+'_aircrafts_resnet34_epoch_1k'+'.pt'
# OPTIMIZER_PATH = 'models/optimizer'+'_aircrafts_resnet34_epoch_1k'+'.pt'
# ACCURACY_PATH = 'models/accuracy'+'_aircrafts_resnet34_epoch_1k'+'.pt'

# torch.save(model, MODEL_PATH)
# torch.save(subclass_prototypes, PROTOTYPES_PATH)
# torch.save(optimizer, OPTIMIZER_PATH)
# torch.save(accuracy_dic, ACCURACY_PATH)


# ## Extra code for all classes

# In[ ]:


# model.eval()

# val_acc = Averager()

# for j, batch in enumerate(val_loader):
#     x, y = batch
#     x = x.to(device)
#     y = y.to(device)

#     coarse_class = coarseclass_name(y[0])
#     fine_class = subclass_name(y[0])
# #     print(coarse_class, fine_class, y[0])

#     if(coarse_class=='Flowers'):
#         _, z, _, _, _, _  = model(x)

#     elif(coarse_class=='Birds'):
#         _, _, z, _, _, _  = model(x)

#     elif(coarse_class=='Dogs'):
#         _, _, _, z, _, _  = model(x)

#     elif(coarse_class=='Cars'):
#         _, _, _, _, z, _  = model(x)

#     elif(coarse_class=='Aircrafts'):
#         _, _, _, _, _, z  = model(x)

#     logits = distance_metric(z, subclass_prototypes)

#     loss = F.cross_entropy(logits, y)  # y has the correct labels
#     acc = count_acc(logits, y)
#     val_acc.add(acc)
#     print(acc, val_acc.item())

