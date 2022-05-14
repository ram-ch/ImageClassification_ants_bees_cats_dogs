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
from tqdm import tqdm


"""
Read a csv file with test files list
parse the path and perform inference
save the results to a csv file with image, label, prediction
"""

# TODO: Save the probabilities per class in the csv


def perform_test(test_image_dir,saved_model,idx_2_class,test_csv):
    test_df=pd.read_csv(test_csv)
    test_images=list(test_df['image'])
    # TODO: Use albumentation for faster transformations
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # load model
    print("Loading trained model...")
    trained_model=torch.load(saved_model)
    trained_model.eval()
    print("Performing inference...")
    start_time = time.time()
    y_pred=[]
    ants_probs=[]
    bees_porbs=[]
    cats_prob=[]
    dogs_prob=[]
    for i in tqdm(test_images):
        img=os.path.join(test_image_dir,i)
        # Load image
        image = Image.open(img)
        image = data_transforms(image).float()
        image = image.clone().detach()
        image = image.unsqueeze(0)
        image = image.to(device)
        # perform inference        
        output=trained_model(image)
        prob = F.softmax(output, dim=1)
        top_p, top_class = prob.topk(4)  
        # labels and probs
        top_probs=top_p.cpu().detach().numpy()[0]
        top_classes=top_class.cpu().detach().numpy()[0]
        classes=[idx_2_class[str(i)] for i in top_classes]
        probabilities=[round(i*100,2) for i in top_probs]    
        prediction=classes[0]
        score=probabilities[0]
        y_pred.append(prediction)

        
        results=dict(zip(classes,top_probs))    
        ants_probs.append(results['ants'])
        bees_porbs.append(results['bees'])
        cats_prob.append(results['cats'])
        dogs_prob.append(results['dogs'])

        

    print(f"Total time taken for {len(test_images)} : {round(time.time() - start_time, 2)} sec(s)")
    test_df['prediction']=y_pred
    test_df['ants_prob']=ants_probs
    test_df['bees_prob']=bees_porbs
    test_df['cats_prob']=cats_prob
    test_df['dogs_prob']=dogs_prob

    test_df.to_csv('test_result.csv',index=False)

if __name__=='__main__':
    # params
    test_image_dir="data/cats_dogs_bees_ants_125/test"
    saved_model="saved_model/2022_05_13_06_30_11_bestmodel.pth"
    idx_2_class={'0':'ants', '1':'bees', '2':'cats', '3':'dogs'}
    test_csv='test.csv'    
    perform_test(test_image_dir,saved_model,idx_2_class,test_csv)