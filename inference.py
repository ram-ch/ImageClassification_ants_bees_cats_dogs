"""
Perform inference on a single image and display the results
"""

import os
import glob
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import seaborn as sns
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# params
test_image_dir="data/cats_dogs_bees_ants_125/test_samples"
saved_model="saved_model/2022_05_13_06_30_11_bestmodel.pth"
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

def view_result(classes, scores):
    x_pos = [i for i, _ in enumerate(classes)]
    plt.bar(x_pos, scores, color='green')
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.title("Prediction")
    plt.xticks(x_pos, classes)
    for i in range(len(classes)):
            plt.text(i,scores[i]+0.5,scores[i], ha = 'center')
    plt.show()



start_time = time.time()
# test image
img=test_images[-1]
# Load image
image = Image.open(img)
image = data_transforms(image).float()
image = image.clone().detach()
image = image.unsqueeze(0)
image = image.to(device)
# perform inference
print("Performing inference...")
output=trained_model(image)
prob = F.softmax(output, dim=1)
top_p, top_class = prob.topk(4)  
# labels and probs
top_probs=top_p.cpu().detach().numpy()[0]
top_classes=top_class.cpu().detach().numpy()[0]
classes=[idx_2_class[str(i)] for i in top_classes]
probabilities=[round(i*100,2) for i in top_probs]
results=dict(zip(classes,probabilities))    
# show image with predictions
img = plt.imread(img)    
plt.imshow(img)
plt.axis('off')
threshold=60
print(results)
print(f"Inference time : {round(time.time() - start_time,2)} secs")
if any(score > threshold for score in probabilities):
    data="ant: {}% bee: {}% cat: {}% dog: {}%".format(results['ants'],results['bees'],results['cats'],results['dogs'])  
else:
    data="Unknown"
plt.text(x=2, y=-3, s=data,c='red',fontsize=10)
plt.show()
# Show plot of score
view_result(classes, probabilities)  


# TODO: use albumentation for faster preprocessing
# TODO: Try c++ inference script