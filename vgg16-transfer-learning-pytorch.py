
# coding: utf-8

# ## VGG implementation with SVM

# *Python Modules*

# # Note:
# A lot of work here is derivative. Multiple sources have been referred to come up with the architecture and the solution given here though the task as a whole has not been directly used. I will make an effort to refer to the sources these to the end.

# In[46]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import sklearn.svm
from sklearn.model_selection import train_test_split, KFold
import random

plt.ion() 

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
else:
    print("Using CPU")


# ImageFolder loads the data directly from its path. transforms are used to then compose the same into the size needed for vggnet and alexnet. The data is then loaded based on the input size. 

# In[53]:


data_dir = "C:/Users/Umashanker Deekshith/Google Drive/Germany/Uni-Bonn/Semester 3/Deep Learning for VR/Exercise/DeepLearningWS/project/Deep-Learning-Project/src/images"
TRAIN = 'train'
TEST = 'test'

def data_loader(data_dir, TRAIN, TEST, image_crop_size = 224, mini_batch_size = 1 ):
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally. 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in [TRAIN, TEST]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=1,
            shuffle=True, num_workers=1
        )
        for x in [TRAIN, TEST]
    }
    return dataloaders, image_datasets
    
dataloaders, image_datasets = data_loader(data_dir, TRAIN, TEST, image_crop_size = 224, mini_batch_size = 1 )

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, TEST]}

for x in [TRAIN, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
print("Classes: ")
class_names = image_datasets[TRAIN].classes
classification_size = len(image_datasets[TRAIN].classes)
print(image_datasets[TRAIN].classes)
print(classification_size)


# ## Utils
# 
# Some utility function to visualize the dataset and the model's predictions

# In[19]:


# Get a batch of training data
# inputs, classes = next(iter(dataloaders[TRAIN]))
# inputs_test, classes_test = next(iter(dataloaders[TEST]))
# def select_dataset(dataloaders, train, test, number_of_classes):
#     inputs, classes = next(iter(dataloaders[train]))
#     inputs_test, classes_test = next(iter(dataloaders[test]))
#     return inputs, classes, inputs_test, classes_test
# inputs, classes, inputs_test, classes_test = select_dataset(dataloaders, TRAIN, TEST, number_of_classes = 10)


# In[21]:


def set_up_network(net, freeze_training = True, clip_classifier = True, classification_size = 101):
    if net == 'vgg16':
    # Load the pretrained model from pytorch
        network = models.vgg16(pretrained=True)

        # Freeze training for all layers
        # Newly created modules have require_grad=True by default
        if freeze_training:
            for param in network.features.parameters():
                param.require_grad = False

        if clip_classifier:
            features = list(network.classifier.children())[:-5] # Remove last layer
            network.classifier = nn.Sequential(*features) # Replace the model classifier
    
    elif net == 'alexnet':
        network = models.alexnet(pretrained=True)
        if freeze_training:
            for param in network.features.parameters():
                param.require_grad = False
        
        if clip_classifier:
            features = list(network.classifier.children())[:-4] # Remove last layer
            network.classifier = nn.Sequential(*features) # Replace the model classifier
    if classification_size != 1000 and clip_classifier == False:
        num_features = network.classifier[6].in_features
        features = list(network.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, classification_size)]) # Add our layer with 4 outputs
        network.classifier = nn.Sequential(*features) # Replace the model cla
#     print(network)
    return network


# In[67]:


vgg16_nc = set_up_network('vgg16', freeze_training = True)
alex_net_nc = set_up_network('alexnet', freeze_training = True)
vgg16_class_10 = set_up_network('vgg16', freeze_training = False, clip_classifier = False, classification_size=10)
alex_net_class_10 = set_up_network('alexnet', freeze_training = False, clip_classifier = False, classification_size = 10)
vgg16_class_30 = set_up_network('vgg16', freeze_training = False, clip_classifier = False, classification_size=30)
alex_net_class_30 = set_up_network('alexnet', freeze_training = False, clip_classifier = False, classification_size = 30)
vgg16_class_100 = set_up_network('vgg16', freeze_training = False, clip_classifier = False, classification_size=100)
alex_net_class_100 = set_up_network('alexnet', freeze_training = False, clip_classifier = False, classification_size = 100)


# ## Task 1: For SVM on top of clipped VGG and AlexNet

# In[60]:


def get_features(ipnet, train_batches = 10, number_of_classes = 10):

    imgfeatures = []
    imglabels = []
    if classification_size < number_of_classes:
        number_of_classes = classification_size
        print("Input size smaller at:", classification_size,". Adjusting the class to this number")
    selected_classes = random.sample(range(0,classification_size), number_of_classes)
    print("The selected classes are: ",selected_classes)
    for i, data in enumerate(dataloaders[TRAIN]):
        if i % 100 == 0:
            print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)

        # Use half training dataset
        if i > train_batches:
            break

        inputs, labels = data
        if(labels.numpy() not in selected_classes): 
            continue
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        feature = ipnet(inputs)
#         print("The shape of output is: ", feature.shape)
#         print(labels)
        imgfeatures.append(feature.detach().numpy().flatten())
        imglabels.append(labels.detach().numpy())
        del inputs, labels, feature

    return imgfeatures, imglabels


# In[65]:


def fit_features_to_SVM(features, labels, train_batch_size, K=5 ):
#     print("The shape of the class is", classes.shape)
    kf = sklearn.model_selection.KFold(n_splits=K)
    kf.get_n_splits(features)
#     print("The split information is: ", kf)
    scores = []
    features = np.array(features)
    labels = np.array(labels)
    print(features.shape)
    print(labels.shape)

    i=0
    for train, test in kf.split(features):
#     for train, test in kf:
#         print(train)
#         print(test)
        i+=1
        model = sklearn.svm.SVC(C=100)#, C=1, gamma=0)
        model.fit(features[train, :], labels[train].ravel())
        s=model.score(features[test, :], labels[test])
        print(i,"/",K,"The score for this classification is: ", s)
        scores.append(s)
    return np.mean(scores), np.std(scores)

def fit_features_to_SVM_new(features, labels, train_batch_size, K=5 ):
#     print("The shape of the class is", classes.shape)
    # split into a training and testing set
    features = np.array(features)
    labels = np.array(labels)
    scores = []
    for i in range(K):
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=(1/K), random_state=42)
#         print("The testing shape is: ", x_test.shape, y_test.shape)
#         print("The training shape is: ", x_train.shape, y_train.shape)
        model = sklearn.svm.SVC(C=100)#, C=1, gamma=0)
        model.fit(x_train, y_train.ravel())
        s=model.score(x_test, y_test)
        print("The score for this classification is: ", s)
        scores.append(s)
    return np.mean(scores), np.std(scores)


# ## VGG16 implementation with SVM as a classification layer. 
# The batch size and other things can be classified from here.

# In[64]:


train_batch_size = 10
Class_Size = [10, 30]
for class_size in Class_Size:
    imgfeatures_vgg, imglabels_vgg = get_features(vgg16_nc, train_batch_size, number_of_classes = class_size)
    mean_accuracy, sd = fit_features_to_SVM(imgfeatures_vgg,
                                            imglabels_vgg, train_batch_size, K=5 )
    print("The mean and standard deviation of classification for vgg 16 is: ",
          mean_accuracy, sd, "for class size: ", class_size)


# ## Alexnet implementation with SVM as a classification layer. 
# The batch size and other things can be classified from here.

# In[66]:


for class_size in Class_Size:
    imgfeatures_an, imglabels_an = get_features(alex_net_nc, train_batch_size, class_size)
    mean_accuracy, sd = fit_features_to_SVM(imgfeatures_an, imglabels_an, train_batch_size, K=5 )
    print("The mean and standard deviation of classification for alexnet is: ",mean_accuracy, sd)


# vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
torch.save(vgg16.state_dict(), 'VGG16_v2-OCT_Retina_half_dataset.pt')
torch.save(alex_net.state_dict(), 'ALEXNET_v2-OCT_Retina_half_dataset.pt')


# ## Task 2: This one trains on top of the existing pre-trained network.

# ## Loss function
# Here, based on whether label smoothing is needed or not, a different loss function is selected.

# In[75]:


def cal_loss(pred, gold, smoothing = False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=0)

    return loss


# In[76]:


if use_gpu:
    vgg16_class_10.cuda() #.cuda() will move everything to the GPU side
    vgg16_class_30.cuda() #.cuda() will move everything to the GPU side
    vgg16_class_100.cuda() #.cuda() will move everything to the GPU side
    alex_net_class_10.cuda() #.cuda() will move everything to the GPU side
    alex_net_class_30.cuda() #.cuda() will move everything to the GPU side
    alex_netvgg16_class_100.cuda() #.cuda() will move everything to the GPU side

# criterion = nn.CrossEntropyLoss()
criterion = cal_loss
optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# ## Training along with validation.
# Here, a split of 80% for training and 20% for validation is done for cross validation. It otherwise follows the standard training example given in pytorch site.
# 

# In[78]:


def train_model(vgg, criterion, optimizer, scheduler, dataloaders, num_epochs=10, label_smoothing = False, classes = 10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    K = 5
    train_batches = len(dataloaders[TRAIN])
    train_bat = np.ones((train_batches, 1)) # This is a dummy variable as sklearn changed stuff and didn't do it right.
    val_batches = 0.2*train_batches
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
       
        kf = sklearn.model_selection.KFold(n_splits=K)
        kf.get_n_splits(train_bat)

    
        for train, test in kf.split(train_bat):
        
            for i, data in enumerate(dataloaders[TRAIN]):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)

                # Use half training dataset
                if i >= train_batches / 2:
#                 if i >= 1:
                    break
                
                if i not in train:
                    continue
                
                inputs, labels = data

                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = vgg(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                loss_train += loss.item()
#                 loss_train += loss.data[0]
                acc_train += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            print()
            # * 2 as we only used half of the dataset
            avg_loss = loss_train * 2 / (dataset_sizes[TRAIN]*0.8)
            avg_acc = acc_train * 2 / (dataset_sizes[TRAIN]*0.8)

            vgg.train(False)
            vgg.eval()

            for i, data in enumerate(dataloaders[TRAIN]):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

#                 if i >= 1:
#                     break
                if i not in test:
                    continue
                
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda(), requires_grad=True), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)

                optimizer.zero_grad()

                outputs = vgg(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

#                 loss_val += loss.data[0]
                loss_train += loss.item()
                acc_val += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            avg_loss_val = loss_val / (dataset_sizes[TRAIN]*0.2)
            avg_acc_val = acc_val / (dataset_sizes[TRAIN]*0.2)

            print()
            print("Epoch {} result: ".format(epoch))
            print("Avg loss (train): {:.4f}".format(avg_loss))
            print("Avg acc (train): {:.4f}".format(avg_acc))
            print("Avg loss (val): {:.4f}".format(avg_loss_val))
            print("Avg acc (val): {:.4f}".format(avg_acc_val))
            print('-' * 10)
            print()

            if avg_acc_val > best_acc:
                best_acc = avg_acc_val
                best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    return vgg


# In[79]:


def eval_model(vgg, criterion, smoothing_labels = False):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)
#         if i >= 1:
#             break
        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), requires_grad=True), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels, smoothing=smoothing_labels)

#         loss_test += loss.data[0]
        loss_test += loss.item()

        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    avg_loss = loss_test / dataset_sizes[TEST]
    avg_acc = acc_test / dataset_sizes[TEST]
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


# ## Training VGG model

# In[80]:


# Number_Of_Classes = [10, 30, 100]
# vgg16_net = [vgg16_class_10, vgg16_class_30, vgg16_class_100]
Number_Of_Classes = [10]
vgg16_net = [vgg16_class_10]
for i, vgg16 in enumerate(vgg16_net):
    if classification_size < Number_Of_Classes[i]:
        Number_Of_Classes[i] = classification_size
        print("Input size smaller at:", classification_size,". Adjusting the class to this number")
    selected_classes = random.sample(range(0,classification_size), Number_Of_Classes[i])

    vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=2, classes = selected_classes)
    print("Testing the trained model")
    eval_model(vgg16, criterion)
    torch.save(vgg16.state_dict(), "VGG16_v2_"+str(Number_Of_Classes[i])+"-OCT_Retina_half_dataset.pt")


# ## Testing the trained network

# In[ ]:


print("Testing the trained model")
eval_model(vgg16_class_10, criterion)


# ## Training AlexNet

# In[ ]:


alex_net = train_model(alex_net, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=2)
torch.save(alex_net.state_dict(), 'AlexNet_v2-OCT_Retina_half_dataset.pt')


# ## Testing the trained network

# In[ ]:


print("Testing the trained model")
eval_model(alex_net, criterion)


# ## Task 3: Using label smoothing regularisation
# The loss function is updated to include smoothing and is as shown here.

# ## VGG16 with label smoothing

# In[ ]:


vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=2, label_smoothing = True)
torch.save(vgg16.state_dict(), 'VGG16_task3_v2-OCT_Retina_half_dataset.pt')


# In[ ]:


print("Testing the trained model")
eval_model(vgg16, criterion,True)


# ## AlexNet with label smoothing

# In[ ]:


alex_net = train_model(alex_net, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=2, label_smoothing = True)
torch.save(alex_net.state_dict(), 'ALEXNet_Task3_v2-OCT_Retina_half_dataset.pt')


# In[ ]:


print("Testing the trained model")
eval_model(alex_net, criterion,True)

