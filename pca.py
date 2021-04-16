import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from numpy import linalg as LA

from sklearn.decomposition import PCA
import numpy as np
import pickle
import random
import torchvision.models as models

def get_model_param_vec(model):
    # Return the model parameters as a vector

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().reshape(-1).numpy())
    return np.concatenate(vec, 0)


parser = argparse.ArgumentParser(description='P+-BFGS in pytorch')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--n_components', default=10, type=int, metavar='N',
                    help='n_components for PCA') 
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='epochs for PCA') 
parser.add_argument('--params_start', default=0, type=int, metavar='N',
                    help='param_start for PCA') 
parser.add_argument('--params_end', default=300, type=int, metavar='N',
                    help='param_end for PCA') 
args = parser.parse_args()

model = torch.nn.DataParallel(models.__dict__[args.arch]())
model.cuda()

data = []
for i in range(args.params_start, args.params_end):
    # if (i > 90 and i % 3 != 0): continue
    f = './save_' + args.arch + '/' + str(i) + '.pt'
    # print (f)
    
    # model.load_state_dict(torch.load('a.pt', map_location="cuda:0"))
    model.load_state_dict(torch.load(f, map_location="cuda:0"))
    # model = torch.nn.DataParallel(model).cuda()
    data.append(get_model_param_vec(model))
data = np.array(data)


W = data
print (W.shape)
pca = PCA(n_components=args.n_components)
pca.fit_transform(W)
print(pca.explained_variance_ratio_)
print (np.array(pca.explained_variance_ratio_).sum())
P = np.array(pca.components_)
print ('P:', P.shape)

print (P.shape)
with open(args.arch + '_epochs' + str(args.epochs) + '_components' + str(args.n_components)+'_from' + str(args.params_start) + 'to'+ str(args.params_end) + '.txt', 'wb') as file:
    pickle.dump(P, file, protocol = 4)
print ('Done!')
