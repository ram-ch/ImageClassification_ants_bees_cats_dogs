import os
import sys
import time
import copy
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import textwrap, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# params
test_image_dir="C:/codebase/ComputerVision/data/sample_test_data"
saved_model="saved_model/2022_01_28_05_33_34_bestmodel.pth"
idx_2_class={'0':'ants', '1':'bees', '2':'cats', '3':'dogs'}

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.clone().detach()
    image = image.unsqueeze(0)
    return image

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# load model
trained_model=torch.load(saved_model)
trained_model.eval()

test_images=glob.glob(test_image_dir+"/*.jpg")
# create figure
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 4

for i,img in enumerate(test_images):
    # Load image
    image=image_loader(data_transforms, img)
    output=trained_model(image)
    prob = F.softmax(output, dim=1)
    top_p, top_class = prob.topk(4)  
    
    top_probs=top_p.detach().numpy()[0]
    top_classes=top_class.detach().numpy()[0]
    classes=[idx_2_class[str(i)] for i in top_classes]
    probabilities=[round(i*100,2) for i in top_probs]
    results=dict(zip(classes,probabilities))
    

    img = plt.imread(img)
    fig.add_subplot(rows, columns, i+1)    
    # showing image
    plt.imshow(img)
    plt.axis('off')
    data="ant: {}% bee: {}% cat: {}% dog: {}%".format(results['ants'],results['bees'],results['cats'],results['dogs'])    
    plt.text(x=2, y=-3, s=data,c='red',fontsize=0.5)
  
plt.show()

# TODO: Run inference on all the images in validation
# TODO: Get the probabilities
# TODO: Compute TPR, FPR
# TODO: Plot the auroc for different thresholds