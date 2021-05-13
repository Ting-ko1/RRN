import torch
import torchvision
from torch import nn
import math
import random
import matplotlib.pylab as plt 
import pylab
import numpy as np


class algorithm(nn.Module):
    def __init__(self):
        super (algorithm, self).__init__()


    def forward(self, x, eps):  
        torch.save(x.cpu(), './non-noise.pt')       #original image
        x=RRN(x,eps)
        torch.save(x.cpu(), './noise.pt')           #noised image
        return x




def RRN(data,eps):
    p=(math.e**eps)/(1+math.e**eps)
    line=data.size()[1]
    s=data.size()[2]
    for i in range (data.size()[0]):
        #print(data)
        laplace_noise=np.random.laplace(size=(line,s,s))              #mnist:5*5   cifar:6*6
        p_noise=np.random.choice([0,1],p=[p,1-p],size=(line,s,s))
        noise=np.multiply(laplace_noise,p_noise)
        #print(noise)
        noise=torch.tensor(noise)
        noise_data=data[i]+noise.cuda()
        data[i]=noise_data.clamp(min=0,max=1)
    return data


