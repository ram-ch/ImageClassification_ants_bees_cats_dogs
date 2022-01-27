import os
import sys
import time
import copy
import glob
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
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from tensorboardX import SummaryWriter


writer = SummaryWriter('runs/image_classifier')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Params
learning_rate=0.001
momentum=0.9
step_size=7
gamma=0.1
num_epochs=25
batch_size=5
mean=np.array([0.45,0.456,0.406])
std=np.array([0.229, 0.224, 0.225])

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
                               ])
}

# import data
data_dir="C:\codebase\data1"
sets=['train','val']
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in sets}
dataloaders={x:torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in sets}
images, labels = iter(dataloaders['train']).next() # sample input
dataset_sizes={x:len(image_datasets[x]) for x in sets}
class_names=image_datasets['train'].classes
print(class_names)


def view_samples(sample_images): 
    # TODO: Get sample images from the train data folders
    label_wrap_length=50
    label_font_size=8
    fig=plt.figure()
    for i,sample in enumerate(sample_images):
        print(sample)
        image=Image.open(sample).resize((256,256),Image.LANCZOS)
        plt.subplot(1, len(sample_images), i+1)
        plt.imshow(image)
        title=os.path.basename(sample).split('.')[0]
        title=textwrap.wrap(title, label_wrap_length)
        title="\n".join(title)
        plt.title(title, fontsize=label_font_size)
        plt.xticks([])
        plt.yticks([])
        # plt.tight_layout()
    # plt.show()
    writer.add_figure('TrainingImages/sample', fig, global_step=0)
    writer.flush()
    # writer.close()
    return

def view_confusionmatrix(predictions, actuals, phase,global_step):  
    # TODO: Make this dict global
    class_2_label={'0':'ants', '1':'bees', '2':'cats', '3':'dogs'}
    predictions=[class_2_label[str(i)] for i in predictions]
    actuals=[class_2_label[str(i)] for i in actuals]
    cf_matrix = confusion_matrix(predictions, actuals)
    df_cm = pd.DataFrame(cf_matrix, columns=np.unique(actuals), index = np.unique(actuals))    
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    fig=plt.figure(figsize = (5,5))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, cmap="Blues", annot=True,fmt='g',annot_kws={"size": 16})
    # plt.show()
    writer.add_figure(f'ConfusionMatrix/{phase}', fig, global_step=global_step)
    writer.flush()
    writer.close()
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


                # forward
                # track history if only in train
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
            view_confusionmatrix(prediction_labels, actual_labels,phase,epoch)

            if phase=='train':
                scheduler.step()
                writer.add_scalars('Loss/Train',{'Loss/Train': epoch_loss},epoch)
                writer.add_scalars('Accuracy/Train',{'Accuracy/Train':epoch_acc},epoch)
                writer.flush()

            elif phase=='val':
                writer.add_scalars('Loss/Val',{'Loss/Val':epoch_loss},epoch)
                writer.add_scalars('Accuracy/Val',{'Accuracy/Val':epoch_acc},epoch)
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


# Convnet as fized feature extractor
# here, we need to freeze all the network except the final layer
# we need to set required_grads==False to freeze all the parameters so the gradients are not computed in backward()
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
view_samples(sample_images=glob.glob("C:/codebase/Pytorch/sample_data/*.jpg"))
# Begin train
model=train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
# Begin Test
writer.close()