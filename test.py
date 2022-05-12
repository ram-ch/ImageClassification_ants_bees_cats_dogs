"""
# In Progress
Run test on a sample of images, compute
Confusion matrix
ROC curves,
F1 score,
map

Save the results to jinja templates
"""
def get_predictions(loader,img,trained_model):
    image = Image.open(img)
    image = loader(image).float()
    image = image.clone().detach()
    image = image.unsqueeze(0)
    output = trained_model(image)
    prob = F.softmax(output, dim=1)
    top_p, top_class = prob.topk(num_classes)  
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
        # plt.imshow(img)
        plt.savefig('test_samples.png')
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
