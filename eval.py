import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from torchvision.utils import save_image

import torchvision.models as models

from arguments import args
       
from jtt_data import valid_loader, valid_blond_loader, valid_blond_men_loader, valid_blond_women_loader, valid_men_loader, valid_women_loader, valid_brunette_loader, valid_brunette_men_loader, valid_brunette_women_loader, test_loader, test_blond_men_loader, test_blond_women_loader, test_brunette_men_loader, test_brunette_women_loader, test_men_loader, test_women_loader  
from jtt_data import custom_transform, CelebaDataset       


# Hyperparameters
DEVICE = 'cuda:0'
BASE_LR = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MOMENTUM = args.momentum
WEIGHT_DECAY = args.weight_decay
UPSAMPLE = args.upsample
STAGE_1_EPOCH = args.stage_1_epoch
RUN = args.run

STAGE_1_PATH_BLOND = 'saved_models/run_%s/bs_128_epochs_50_lr_0.0001_wd_0.0001_class_balanced/'%(RUN)
STAGE_1_PATH_BRUNETTE = 'saved_models/run_%s/bs_128_epochs_50_lr_1e-05_wd_0.1_class_balanced/'%(RUN)
STAGE_2_PATH = 'saved_models/run_%s/bs_%s_epochs_%s_lr_%s_wd_%s/epoch_%s/stage_2/incorrect_upsampled_%s_bs_%s_epochs_%s_lr_%s_wd_%s/'%(RUN, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY, STAGE_1_EPOCH, UPSAMPLE, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY)
SAVE_PATH = STAGE_2_PATH + 'plots/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

blond_mses = np.loadtxt(STAGE_1_PATH_BLOND + 'statistics/blond_mses.txt') 
BLOND_STAGE_1_MODEL = np.argmax(blond_mses) + 1
df_blond = pd.read_csv(STAGE_1_PATH_BLOND + 'statistics/df_blond.csv')
df_blond_noisy_men = df_blond[df_blond['mistakes_epoch_%s'%(BLOND_STAGE_1_MODEL)] == 1]
df_blond_noisy_women = df_blond[df_blond['mistakes_epoch_%s'%(BLOND_STAGE_1_MODEL)] == 0]
blond_base_model = models.resnet50(pretrained=True)
d = blond_base_model.fc.in_features
blond_base_model.fc = nn.Linear(d, 1)
if torch.cuda.device_count() > 1:
    blond_base_model = nn.DataParallel(blond_base_model, device_ids=[0])
blond_base_model.to(DEVICE)
load_path = STAGE_1_PATH_BLOND + 'epoch_%s/epoch_%s.pt'%(BLOND_STAGE_1_MODEL, BLOND_STAGE_1_MODEL)
print('Loading Blond Base Model Checkpoint: ', load_path)
checkpoint = torch.load(load_path)
blond_base_model.load_state_dict(checkpoint['model_state_dict'])
blond_base_model.eval()

brunette_mses = np.loadtxt(STAGE_1_PATH_BRUNETTE + 'statistics/brunette_mses.txt')
BRUNETTE_STAGE_1_MODEL = np.argmax(brunette_mses) + 1
df_brunette = pd.read_csv(STAGE_1_PATH_BRUNETTE + 'statistics/df_brunette.csv')
df_brunette_noisy_men = df_brunette[df_brunette['mistakes_epoch_%s'%(BRUNETTE_STAGE_1_MODEL)] == 0]
df_brunette_noisy_women = df_brunette[df_brunette['mistakes_epoch_%s'%(BRUNETTE_STAGE_1_MODEL)] == 1]
brunette_base_model = models.resnet50(pretrained=True)
d = brunette_base_model.fc.in_features
brunette_base_model.fc = nn.Linear(d, 1)
if torch.cuda.device_count() > 1:
    brunette_base_model = nn.DataParallel(brunette_base_model, device_ids=[0])
brunette_base_model.to(DEVICE)
load_path = STAGE_1_PATH_BRUNETTE + 'epoch_%s/epoch_%s.pt'%(BRUNETTE_STAGE_1_MODEL, BRUNETTE_STAGE_1_MODEL)
print('Loading Brunette Base Model Checkpoint: ', load_path)
checkpoint = torch.load(load_path)
brunette_base_model.load_state_dict(checkpoint['model_state_dict'])
brunette_base_model.eval()

df_noisy_men = df_blond_noisy_men.append(df_brunette_noisy_men, ignore_index=True)
df_noisy_women = df_blond_noisy_women.append(df_brunette_noisy_women, ignore_index=True)

df_noisy_men_dataset = CelebaDataset(df=df_noisy_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
df_noisy_men_loader = DataLoader(dataset=df_noisy_men_dataset,
                  batch_size=1024,
                  shuffle=True,
                  num_workers=4, 
                  pin_memory=True)
df_noisy_women_dataset = CelebaDataset(df=df_noisy_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
df_noisy_women_loader = DataLoader(dataset=df_noisy_women_dataset,
                  batch_size=1024,
                  shuffle=True,
                  num_workers=4, 
                  pin_memory=True)     
                  
def demographic_parity(net, majority_loader, minority_loader, device):
    majority_positive_pred, minority_positive_pred = 0.0, 0.0
    majority_num_examples, minority_num_examples = 0.0, 0.0
    
    with torch.no_grad():
        for i, (features, _, _, _) in enumerate(majority_loader):
            features = features.to(DEVICE).float()
            logits = net(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5
            majority_num_examples += features.size(0)
            majority_positive_pred += (predicted_labels == 1).sum()
            
    with torch.no_grad():
        for i, (features, _, _, _) in enumerate(minority_loader):
            features = features.to(DEVICE).float()
            logits = net(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5
            minority_num_examples += features.size(0)
            minority_positive_pred += (predicted_labels == 1).sum()
            
    dp_gap = abs(((majority_positive_pred/majority_num_examples) - (minority_positive_pred/minority_num_examples))*100)
    dp_gap = dp_gap.item()
    
    return dp_gap     

def compute_loss_accuracy(net, loader, device):
    correct_pred, num_examples = 0.0, 0.0
    running_loss = 0.0
    cost_fn = torch.nn.BCELoss()

    with torch.no_grad():
        for i, (features, _, attributes, targets) in enumerate(loader):
            
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()

            logits = net(features)
            probas = torch.sigmoid(logits)

            primary_loss = cost_fn(probas, targets)
            running_loss += primary_loss.item()*targets.size(0)

            predicted_labels = probas > 0.5
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    loss = running_loss/num_examples
    accuracy = (correct_pred/num_examples) * 100
    accuracy = accuracy.item()
    
    return loss, accuracy     
    
def compute_correct_incorrect_loss(base_net, net, loader, device):
    correct_num_examples, incorrect_num_examples = 0.0, 0.0
    correct_pred, incorrect_pred = 0.0, 0.0
    correct_running_loss, incorrect_running_loss = 0.0, 0.0
    cost_fn = torch.nn.BCELoss()

    with torch.no_grad():
        for i, (features, _, _, targets) in enumerate(loader):
            
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()
            logits = base_net(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5 
            
            correct = torch.where(predicted_labels == targets)[0]            
            incorrect = torch.where(predicted_labels != targets)[0]            
            correct_features = features[correct]
            correct_targets = targets[correct]
            incorrect_features = features[incorrect]
            incorrect_targets = targets[incorrect]
            
            logits = net(correct_features)
            probas = torch.sigmoid(logits)
            primary_loss = cost_fn(probas, correct_targets)
            correct_running_loss += primary_loss.item()*correct_targets.size(0)
            predicted_labels = probas > 0.5
            correct_num_examples += correct_targets.size(0)
            correct_pred += (predicted_labels == correct_targets).sum()
            
            logits = net(incorrect_features)
            probas = torch.sigmoid(logits)
            primary_loss = cost_fn(probas, incorrect_targets)
            incorrect_running_loss += primary_loss.item()*incorrect_targets.size(0)
            predicted_labels = probas > 0.5
            incorrect_num_examples += incorrect_targets.size(0)
            incorrect_pred += (predicted_labels == incorrect_targets).sum()
            
    correct_loss = correct_running_loss/correct_num_examples
    correct_accuracy = (correct_pred/correct_num_examples) * 100
    correct_accuracy = correct_accuracy.item()
    
    incorrect_loss = incorrect_running_loss/incorrect_num_examples
    incorrect_accuracy = (incorrect_pred/incorrect_num_examples) * 100
    incorrect_accuracy = incorrect_accuracy.item()
    
    return correct_loss, correct_accuracy, incorrect_loss, incorrect_accuracy   
    
def eval_validation(model, device):
    l1, s1 = compute_loss_accuracy(model, valid_blond_men_loader, DEVICE)
    l2, s2 = compute_loss_accuracy(model, valid_blond_women_loader, DEVICE)
    l3, s3 = compute_loss_accuracy(model, valid_brunette_men_loader, DEVICE)
    l4, s4 = compute_loss_accuracy(model, valid_brunette_women_loader, DEVICE)
    return l1, l2, l3, l4, s1, s2, s3, s4

def eval_test(model, device):        
    l1, s1 = compute_loss_accuracy(model, test_blond_men_loader, DEVICE)
    l2, s2 = compute_loss_accuracy(model, test_blond_women_loader, DEVICE)
    l3, s3 = compute_loss_accuracy(model, test_brunette_men_loader, DEVICE)
    l4, s4 = compute_loss_accuracy(model, test_brunette_women_loader, DEVICE)
    return l1, l2, l3, l4, s1, s2, s3, s4

valid_blond_men_loss, valid_blond_women_loss, valid_brunette_men_loss, valid_brunette_women_loss = [], [], [], []
valid_blond_men_acc, valid_blond_women_acc, valid_brunette_men_acc, valid_brunette_women_acc = [], [], [], []

test_blond_men_loss, test_blond_women_loss, test_brunette_men_loss, test_brunette_women_loss = [], [], [], []
test_blond_men_acc, test_blond_women_acc, test_brunette_men_acc, test_brunette_women_acc = [], [], [], []

correct_losses_blond, correct_accuracies_blond, incorrect_losses_blond, incorrect_accuracies_blond = [], [], [], []
correct_losses_brunette, correct_accuracies_brunette, incorrect_losses_brunette, incorrect_accuracies_brunette = [], [], [], []

dp_gaps, valid_dp_gaps, test_dp_gaps = [], [], []

valid_avg_loss, valid_avg_acc = [], []
test_avg_loss, test_avg_acc = [], []

for epoch in range(EPOCHS):
    model = models.resnet50(pretrained=True)
    d = model.fc.in_features
    model.fc = nn.Linear(d, 1)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0])
    model.to(DEVICE)    
    load_path = STAGE_2_PATH + 'epochs/epoch_%s/epoch_%s.pt'%(epoch+1, epoch+1)
    print('Loading Model Checkpoint: ', load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    l1, l2, l3, l4, s1, s2, s3, s4 = eval_validation(model, DEVICE)
    valid_blond_men_loss.append(l1)
    valid_blond_women_loss.append(l2)
    valid_brunette_men_loss.append(l3)
    valid_brunette_women_loss.append(l4)
    valid_blond_men_acc.append(s1)
    valid_blond_women_acc.append(s2)
    valid_brunette_men_acc.append(s3)
    valid_brunette_women_acc.append(s4)
    
    l1, l2, l3, l4, s1, s2, s3, s4 = eval_test(model, DEVICE)
    test_blond_men_loss.append(l1)
    test_blond_women_loss.append(l2)
    test_brunette_men_loss.append(l3)
    test_brunette_women_loss.append(l4)
    test_blond_men_acc.append(s1)
    test_blond_women_acc.append(s2)
    test_brunette_men_acc.append(s3)
    test_brunette_women_acc.append(s4)
    
    correct_loss_blond, correct_accuracy_blond, incorrect_loss_blond, incorrect_accuracy_blond = compute_correct_incorrect_loss(blond_base_model, model, valid_blond_loader, DEVICE)
    correct_loss_brunette, correct_accuracy_brunette, incorrect_loss_brunette, incorrect_accuracy_brunette = compute_correct_incorrect_loss(brunette_base_model, model, valid_brunette_loader, DEVICE) 
    correct_losses_blond.append(correct_loss_blond) 
    correct_accuracies_blond.append(correct_accuracy_blond)
    incorrect_losses_blond.append(incorrect_loss_blond)
    incorrect_accuracies_blond.append(incorrect_accuracy_blond)
    correct_losses_brunette.append(correct_loss_brunette)
    correct_accuracies_brunette.append(correct_accuracy_brunette)
    incorrect_losses_brunette.append(incorrect_loss_brunette)
    incorrect_accuracies_brunette.append(incorrect_accuracy_brunette)
    
    v_l, v_a = compute_loss_accuracy(model, valid_loader, DEVICE)
    valid_avg_loss.append(v_l)
    valid_avg_acc.append(v_a)
    
    t_l, t_a = compute_loss_accuracy(model, test_loader, DEVICE)
    test_avg_loss.append(t_l)
    test_avg_acc.append(t_a)
    
    dp_gap = demographic_parity(model, df_noisy_men_loader, df_noisy_women_loader, DEVICE)
    valid_dp_gap = demographic_parity(model, valid_men_loader, valid_women_loader, DEVICE)
    test_dp_gap = demographic_parity(model, test_men_loader, test_women_loader, DEVICE)
    dp_gaps.append(dp_gap)
    valid_dp_gaps.append(valid_dp_gap)
    test_dp_gaps.append(test_dp_gap)
    
    if (epoch+1)%1 == 0:

        np.savetxt(SAVE_PATH + "valid_blond_men_loss.txt", valid_blond_men_loss)
        np.savetxt(SAVE_PATH + "valid_blond_women_loss.txt", valid_blond_women_loss)
        np.savetxt(SAVE_PATH + "valid_brunette_men_loss.txt", valid_brunette_men_loss)
        np.savetxt(SAVE_PATH + "valid_brunette_women_loss.txt", valid_brunette_women_loss)
        np.savetxt(SAVE_PATH + "valid_blond_men_acc.txt", valid_blond_men_acc)
        np.savetxt(SAVE_PATH + "valid_blond_women_acc.txt", valid_blond_women_acc)
        np.savetxt(SAVE_PATH + "valid_brunette_men_acc.txt", valid_brunette_men_acc)
        np.savetxt(SAVE_PATH + "valid_brunette_women_acc.txt", valid_brunette_women_acc)
        
        np.savetxt(SAVE_PATH + "test_blond_men_loss.txt", test_blond_men_loss) 
        np.savetxt(SAVE_PATH + "test_blond_women_loss.txt", test_blond_women_loss)  
        np.savetxt(SAVE_PATH + "test_brunette_men_loss.txt", test_brunette_men_loss) 
        np.savetxt(SAVE_PATH + "test_brunette_women_loss.txt", test_brunette_women_loss)
        np.savetxt(SAVE_PATH + "test_blond_men_acc.txt", test_blond_men_acc) 
        np.savetxt(SAVE_PATH + "test_blond_women_acc.txt", test_blond_women_acc)  
        np.savetxt(SAVE_PATH + "test_brunette_men_acc.txt", test_brunette_men_acc) 
        np.savetxt(SAVE_PATH + "test_brunette_women_acc.txt", test_brunette_women_acc)

        np.savetxt(SAVE_PATH + "correct_losses_blond.txt", correct_losses_blond)
        np.savetxt(SAVE_PATH + "correct_accuracies_blond.txt", correct_accuracies_blond)
        np.savetxt(SAVE_PATH + "incorrect_losses_blond.txt", incorrect_losses_blond)
        np.savetxt(SAVE_PATH + "incorrect_accuracies_blond.txt", incorrect_accuracies_blond)
        np.savetxt(SAVE_PATH + "correct_losses_brunette.txt", correct_losses_brunette)
        np.savetxt(SAVE_PATH + "correct_accuracies_brunette.txt", correct_accuracies_brunette)
        np.savetxt(SAVE_PATH + "incorrect_losses_brunette.txt", incorrect_losses_brunette) 
        np.savetxt(SAVE_PATH + "incorrect_accuracies_brunette.txt", incorrect_accuracies_brunette)
        
        np.savetxt(SAVE_PATH + "dp_gap.txt", dp_gaps)
        np.savetxt(SAVE_PATH + "valid_dp_gap.txt", valid_dp_gaps)
        np.savetxt(SAVE_PATH + "test_dp_gap.txt", test_dp_gaps)

        np.savetxt(SAVE_PATH + "valid_avg_loss.txt", valid_avg_loss)
        np.savetxt(SAVE_PATH + "valid_avg_acc.txt", valid_avg_acc)
        np.savetxt(SAVE_PATH + "test_avg_loss.txt", test_avg_loss)
        np.savetxt(SAVE_PATH + "test_avg_acc.txt", test_avg_acc)
        
