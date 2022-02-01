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
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score,f1_score, log_loss, roc_curve, auc

import torch.nn.functional as F
from tensorboardX import SummaryWriter


writer = SummaryWriter('runs/image_classifier')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths
data_dir="C:/codebase/ComputerVision/data/ants_bees_cats_dogs"
sample_train_dir="C:/codebase/ComputerVision/data/sample_train_data"
sample_test_dir="C:/codebase/ComputerVision/data/sample_test_data"
test_image_dir="C:/codebase/ComputerVision/data/ants_bees_cats_dogs/val"

# Params
learning_rate=0.001
momentum=0.9
step_size=7
gamma=0.1
num_epochs=1
batch_size=5
mean=np.array([0.45,0.456,0.406])
std=np.array([0.229, 0.224, 0.225])
idx_2_class={'0':'ants', '1':'bees', '2':'cats', '3':'dogs'}
class_2_idx={'ants': 0, 'bees': 1,'cats':2,'dogs':3}

# Data transformers
data_transforms={
    'train': transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean,std)
                                ]),
    'val': transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean,std)
                               ]),
    'test': transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
}

# import data
sets=['train','val']
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in sets}
dataloaders={x:torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in sets}
images, labels = iter(dataloaders['train']).next() # sample input
dataset_sizes={x:len(image_datasets[x]) for x in sets}
class_names=image_datasets['train'].classes


def view_samples(sample_images): 
    # TODO: Get sample images from the train data folders
    label_wrap_length=50
    label_font_size=8
    fig=plt.figure()
    for i,sample in enumerate(sample_images):
        image=Image.open(sample).resize((256,256),Image.LANCZOS)
        plt.subplot(1, len(sample_images), i+1)
        plt.imshow(image)
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


def get_predictions(loader,img,trained_model):
    image = Image.open(img)
    image = loader(image).float()
    image = image.clone().detach()
    image = image.unsqueeze(0)
    output = trained_model(image)
    prob = F.softmax(output, dim=1)
    top_p, top_class = prob.topk(4)  
    # get labels and scores
    top_probs=top_p.detach().numpy()[0]
    top_classes=top_class.detach().numpy()[0]
    return top_probs,top_classes


def view_sample_results(trained_model,sample_test_dir):
    test_images=glob.glob(sample_test_dir+"/*.jpg")
    # create figure
    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 4
    for i,img in enumerate(test_images):
        # Load image
        top_probs, top_classes = get_predictions(data_transforms['test'], img, trained_model)
        classes=[idx_2_class[str(i)] for i in top_classes]
        probabilities=[round(i*100,2) for i in top_probs]
        results=dict(zip(classes,probabilities))        
        img = plt.imread(img)
        fig.add_subplot(rows, columns, i+1)    
        # showing image
        plt.imshow(img)
        plt.axis('off')
        data="ant: {}%\nbee: {}%\ncat: {}%\ndog: {}%".format(results['ants'],results['bees'],results['cats'],results['dogs'])    
        plt.text(x=2, y=-3, s=data,c='red',fontsize=7)
    writer.add_figure('Testimages/sample', fig, global_step=0)
    writer.flush()


def view_auc_roc(trained_model,test_image_dir):   
    test_images=glob.glob(test_image_dir+"/*/*.jpg") 
    final_predictions=[]
    ground_truth=[]
    j=0
    print("Performing inference on test images...")
    for i,img in enumerate(test_images):
        ground_truth.append(img.split("\\")[1])
        # Load image
        top_probs, top_classes = get_predictions(data_transforms['test'], img, trained_model)    
        classes=[idx_2_class[str(i)] for i in top_classes]
        probabilities=[round(i*100,2) for i in top_probs]
        results=dict(zip(classes,top_probs))
        final_predictions.append(results)

    ground_truth=[class_2_idx[i] for i in ground_truth]
    results_df=pd.DataFrame.from_dict(final_predictions)
    # create figure
    fig = plt.figure(figsize=(10, 7))
    for i,c in enumerate(class_names):
        fpr, tpr, thresh = roc_curve(ground_truth, results_df[c].tolist(), pos_label=i)
        auc_val=round(auc(fpr,tpr),4)
        plt.plot(fpr,tpr,label=f'AUC {c}: {auc_val}')
        plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic - Multiclass")
        plt.legend(loc="lower right")    

    writer.add_figure('AUC-ROC', fig, global_step=0)
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
            for inputs, labels in dataloaders[phase]:
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


# Freeze all the network except the final layer 
# Set required_grads==False to freeze all the parameters so the gradients are not computed in backward()
model=torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad=False

writer.add_graph(model, images)
writer.flush()

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs=model.fc.in_features
model.fc=nn.Linear(num_ftrs,len(class_names))
model=model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=momentum)
scheduler=lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# view train samples
view_samples(sample_images=glob.glob(sample_train_dir+"/*.jpg"))
# Begin train
model=train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
# save model
date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S")
save_path=f"saved_model/{date}_bestmodel.pth"
torch.save(model, save_path)

# Begin Test 
saved_model="saved_model/2022_01_28_05_33_34_bestmodel.pth"
# load model
trained_model=torch.load(saved_model)
trained_model.eval()
print("Loaded saved model...")
# View sample results
view_sample_results(trained_model,sample_test_dir)
# Perform full scale testing and view roc curves
view_auc_roc(trained_model,test_image_dir)
writer.close()