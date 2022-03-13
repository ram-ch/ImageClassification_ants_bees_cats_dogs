import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
# from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# params
test_image_dir="C:/codebase/ComputerVision/data/sample_test_data"
saved_model="saved_model/2022_01_28_05_33_34_bestmodel.pth"
idx_2_class={'0':'ants', '1':'bees', '2':'cats', '3':'dogs'}


data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# load model
print("Loading trained model...")
trained_model=torch.load(saved_model)
trained_model.eval()

test_images=glob.glob(test_image_dir+"/*.jpg")
# create figure
fig = plt.figure(figsize=(10, 7))
for i,img in enumerate(test_images):
    # Load image
    image = Image.open(img)
    image = data_transforms(image).float()
    image = image.clone().detach()
    image = image.unsqueeze(0)
    # perform inference
    print("Performing inference...")
    output=trained_model(image)
    prob = F.softmax(output, dim=1)
    top_p, top_class = prob.topk(4)  
    # labels and probs
    top_probs=top_p.detach().numpy()[0]
    top_classes=top_class.detach().numpy()[0]
    classes=[idx_2_class[str(i)] for i in top_classes]
    probabilities=[round(i*100,2) for i in top_probs]
    results=dict(zip(classes,probabilities))    
    # show predictions
    img = plt.imread(img)    
    plt.imshow(img)
    plt.axis('off')
    data="ant: {}% bee: {}% cat: {}% dog: {}%".format(results['ants'],results['bees'],results['cats'],results['dogs'])    
    plt.text(x=2, y=-3, s=data,c='red',fontsize=10)
    plt.show()


# TODO: How to identify negative samples
# TODO: Add inference time
# TODO: Try c++ inference script