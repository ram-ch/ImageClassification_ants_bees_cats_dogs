# for sys utils
import os
import sys
import time
import copy
import glob
from datetime import datetime
# for data
import numpy as np
import pandas as pd
# for images & viz
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import textwrap, os
import cv2
# for model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import resnet18, resnet101, vgg16
from torchvision import datasets, models, transforms
# for metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score,  roc_curve, auc
# for augmentation
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/image_classifier')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        super(ImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index]*len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])        
        image_filepath=os.path.join(root_and_dir, img_file)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label



def view_samples(sample_images): 
    label_wrap_length=50
    label_font_size=8
    fig=plt.figure()
    for i,sample in enumerate(sample_images):
        image=Image.open(sample).resize((256,256),Image.LANCZOS)
        plt.subplot(1, len(sample_images), i+1)
        plt.imshow(image)
        plt.savefig('train_samples.png')
        title=os.path.basename(sample).split('.')[0]
        title=textwrap.wrap(title, label_wrap_length)
        title="\n".join(title)
        plt.title(title, fontsize=label_font_size)
        plt.xticks([])
        plt.yticks([])
    writer.add_figure('TrainingImages/sample', fig, global_step=0)
    writer.flush()
    return

def view_confusionmatrix(predictions, actuals, phase,global_step):  
    predictions=[idx_2_class[str(i)] for i in predictions]
    actuals=[idx_2_class[str(i)] for i in actuals]
    cf_matrix = confusion_matrix(predictions, actuals)
    df_cm = pd.DataFrame(cf_matrix, columns=np.unique(actuals), index = np.unique(actuals))    
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    fig=plt.figure(figsize = (5,5))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, cmap="Blues", annot=True,fmt='g',annot_kws={"size": 16})
    writer.add_figure(f'ConfusionMatrix/{phase}', fig, global_step=global_step)
    writer.flush()
    return

def view_precision_recall_f1(predictions, actuals,phase,epoch):    
    predictions=[idx_2_class[str(i)] for i in predictions]
    actuals=[idx_2_class[str(i)] for i in actuals]
    precision=precision_score(predictions,actuals,average=None)
    recall=recall_score(predictions, actuals,average=None)
    f1score=f1_score(predictions, actuals,average=None)
    writer.add_scalars('Precision',{'ants':precision[0],'bees':precision[1],'cats':precision[2],'dogs':precision[3]}, epoch)
    writer.add_scalars('Recall',{'ants':recall[0],'bees':recall[1],'cats':recall[2],'dogs':recall[3]}, epoch)
    writer.add_scalars('f1score',{'ants':f1score[0],'bees':f1score[1],'cats':f1score[2],'dogs':f1score[3]}, epoch)
    writer.flush()
    return


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0            
            prediction_labels=[]
            actual_labels=[]
            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward - track history if only in train
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                pr=list(preds.cpu().detach().numpy())
                act=list(labels.data.cpu().detach().numpy())
                
                prediction_labels.extend(pr)
                actual_labels.extend(act)                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} | {phase} Accuracy: {epoch_acc:.4f}")

            # Compute classification metrics
            view_confusionmatrix(prediction_labels, actual_labels,phase,epoch)             
            
            if phase=='train':
                scheduler.step()
                writer.add_scalars('Loss/Train',{'Loss/Train': epoch_loss},epoch)
                writer.add_scalars('Accuracy/Train',{'Accuracy/Train':epoch_acc},epoch)
                writer.flush()

            elif phase=='val':
                writer.add_scalars('Loss/Val',{'Loss/Val':epoch_loss},epoch)
                writer.add_scalars('Accuracy/Val',{'Accuracy/Val':epoch_acc},epoch)
                view_precision_recall_f1(prediction_labels, actual_labels,phase,epoch)
                writer.flush()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:               
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__=='__main__':

    # TODO: Read params from config
    # TODO: Set pretrained model from config
    # TODO: Modularize the model selection, optimizer selection

    # paths
    data_dir="data/cats_dogs_bees_ants_125/"
    sample_train_dir=os.path.join(data_dir,"train_samples")

    # Params
    model='resnet101'
    num_classes=4
    learning_rate=0.001
    momentum=0.9
    step_size=7
    gamma=0.1
    num_epochs=30
    batch_size=40
    optimizer='Adam'
    mean=np.array([0.45,0.456,0.406])
    std=np.array([0.229, 0.224, 0.225])
    idx_2_class={'0':'ants', '1':'bees', '2':'cats', '3':'dogs'}
    class_2_idx={'ants': 0, 'bees': 1,'cats':2,'dogs':3}
    class_names=['ants','bees','cats','dogs']

    # 1. Data preprocession
    # Image augmentations
    transform = A.Compose([A.Resize(width=256,height=256),
                                A.CenterCrop(width=224,height=224),
                                A.Normalize(mean,std),
                                ToTensorV2()])
    # train and val data loader
    sets=['train','val']
    image_datasets={x:ImageFolder(os.path.join(data_dir,x),transform=transform) for x in sets}
    dataloader={x:DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in sets}
    images, labels = iter(dataloader['train']).next() # sample input
    dataset_sizes={x:len(image_datasets[x]) for x in sets}
    # view train samples
    view_samples(sample_images=glob.glob(sample_train_dir+"/*.jpg"))

    # 2. Build models - Using pretrained for transfer learning
    if model=='resnet101':
        model=resnet101(pretrained=True)
    elif model=='vgg16':
        model=resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    # add mdoel graph to TB
    writer.add_graph(model, images)
    writer.flush()
    
    num_ftrs=model.fc.in_features
    model.fc=nn.Linear(num_ftrs,len(class_names))
    model=model.to(device)
    criterion=nn.CrossEntropyLoss()
    if optimizer=="Adam":
        optimizer=optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=momentum)        
    elif optimizer=="SGD":
        optimizer=optim.SGD(model.fc.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)    
    scheduler=lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


    # 3. Train & val 
    model=train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
    # save model
    date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S")
    save_path=f"saved_model/{date}_bestmodel.pth"
    torch.save(model, save_path) # save model