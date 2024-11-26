import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob

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

from jtt_data import valid_loader, valid_blond_loader, valid_blond_men_loader, valid_blond_women_loader, valid_brunette_loader, valid_brunette_men_loader, valid_brunette_women_loader
from jtt_data import valid_blond, valid_blond_men, valid_blond_women, valid_brunette, valid_brunette_men, valid_brunette_women


def calculate_mse_bias(stage_1_net, loader, device):
    correct_num_examples, incorrect_num_examples = 0.0, 0.0
    correct_features_sum, incorrect_features_sum = 0.0, 0.0
    
    with torch.no_grad():
        for i, (features, _, _, targets) in enumerate(loader):
            
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()
            logits = stage_1_net(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5 
            
            correct = torch.where(predicted_labels == targets)[0]          
            incorrect = torch.where(predicted_labels != targets)[0]            
            
            features = features.mul(255).add_(0.5).clamp_(0, 255)#.permute(0, 2, 3, 1)
            correct_features = features[correct]
            correct_targets = targets[correct]
            incorrect_features = features[incorrect]
            incorrect_targets = targets[incorrect]
            
            correct_num_examples += correct_targets.size(0)
            incorrect_num_examples += incorrect_targets.size(0)
            correct_features_sum += torch.sum(correct_features, 0)
            incorrect_features_sum += torch.sum(incorrect_features, 0)

    correct_features_mean = correct_features_sum/correct_num_examples
    incorrect_features_mean = incorrect_features_sum/incorrect_num_examples
    
    mse = ((correct_features_mean - incorrect_features_mean)**2).mean()
    
    return correct_features_mean/255, incorrect_features_mean/255, mse.to('cpu').numpy()   

def difficult_examples(net, df, loader, epoch, device):
    with torch.no_grad():
        for i, (features, names, _, targets) in enumerate(loader):

            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()
                        
            logits = net(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5
             
            incorrect = torch.where(predicted_labels != targets)[0]            
            incorrect = list(incorrect.to('cpu').numpy())
            incorrect_names = [names[i] for i in incorrect]
            
            df.loc[df['image_id'].isin(incorrect_names), 'mistakes_epoch_%s'%(epoch)] = 1
        
    return df 

# Hyperparameters
DEVICE = 'cuda:0'
BASE_LR = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MOMENTUM = args.momentum
WEIGHT_DECAY = args.weight_decay
RUN = args.run

PATH = 'saved_models/run_%s/bs_%s_epochs_%s_lr_%s_wd_%s_class_balanced/'%(RUN, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY)
if not os.path.exists(PATH+'statistics_valid_distribution/'):
    os.makedirs(PATH+'statistics_valid_distribution/')    

mses = []        
blond_mses = []
brunette_mses = []
blond_men_mses = []
blond_women_mses = []
brunette_men_mses = []
brunette_women_mses = []

df_blond = valid_blond.copy()
df_blond_men = valid_blond_men.copy()
df_blond_women = valid_blond_women.copy()
df_brunette = valid_brunette.copy()
df_brunette_men = valid_brunette_men.copy()
df_brunette_women = valid_brunette_women.copy()

for STAGE_1_MODEL in range(1, 51):
    if not os.path.exists(PATH + 'statistics/epoch_%s/'%(STAGE_1_MODEL)):
        os.makedirs(PATH + 'statistics/epoch_%s/'%(STAGE_1_MODEL))    

    stage_1_model = models.resnet50(pretrained=True)
    d = stage_1_model.fc.in_features
    stage_1_model.fc = nn.Linear(d, 1)
    if torch.cuda.device_count() > 1:
        stage_1_model = nn.DataParallel(stage_1_model, device_ids=[0])
    stage_1_model.to(DEVICE)    
    load_path = PATH + 'epoch_%s/epoch_%s.pt'%(STAGE_1_MODEL, STAGE_1_MODEL)
    print('Loading Model Checkpoint: ', load_path) 
    checkpoint = torch.load(load_path) 
    stage_1_model.load_state_dict(checkpoint['model_state_dict'])
    stage_1_model.eval()

    _, _, mse = calculate_mse_bias(stage_1_model, valid_loader, DEVICE)
    mses.append(mse)
    np.savetxt(PATH + "statistics/mses.txt", mses)    
        
    correct_features_mean, incorrect_features_mean, blond_mse = calculate_mse_bias(stage_1_model, valid_blond_loader, DEVICE)
    blond_mses.append(blond_mse)
    np.savetxt(PATH + "statistics/blond_mses.txt", blond_mses)    
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/blond_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/blond_incorrect_features_mean.png'%(STAGE_1_MODEL))
    
    correct_features_mean, incorrect_features_mean, brunette_mse = calculate_mse_bias(stage_1_model, valid_brunette_loader, DEVICE)    
    brunette_mses.append(brunette_mse)
    np.savetxt(PATH + "statistics/brunette_mses.txt", brunette_mses)
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/brunette_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/brunette_incorrect_features_mean.png'%(STAGE_1_MODEL))

    correct_features_mean, incorrect_features_mean, blond_men_mse = calculate_mse_bias(stage_1_model, valid_blond_men_loader, DEVICE)
    blond_men_mses.append(blond_men_mse)
    np.savetxt(PATH + "statistics/blond_men_mses.txt", blond_men_mses)    
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/blond_men_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/blond_men_incorrect_features_mean.png'%(STAGE_1_MODEL))

    correct_features_mean, incorrect_features_mean, blond_women_mse = calculate_mse_bias(stage_1_model, valid_blond_women_loader, DEVICE)
    blond_women_mses.append(blond_women_mse)
    np.savetxt(PATH + "statistics/blond_women_mses.txt", blond_women_mses)    
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/blond_women_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/blond_women_incorrect_features_mean.png'%(STAGE_1_MODEL))

    correct_features_mean, incorrect_features_mean, brunette_men_mse = calculate_mse_bias(stage_1_model, valid_brunette_men_loader, DEVICE)    
    brunette_men_mses.append(brunette_men_mse)
    np.savetxt(PATH + "statistics/brunette_men_mses.txt", brunette_men_mses)
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/brunette_men_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/brunette_men_incorrect_features_mean.png'%(STAGE_1_MODEL))

    correct_features_mean, incorrect_features_mean, brunette_women_mse = calculate_mse_bias(stage_1_model, valid_brunette_women_loader, DEVICE)    
    brunette_women_mses.append(brunette_women_mse)
    np.savetxt(PATH + "statistics/brunette_women_mses.txt", brunette_women_mses)
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/brunette_women_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/brunette_women_incorrect_features_mean.png'%(STAGE_1_MODEL))

    df_blond['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_blond = difficult_examples(stage_1_model, df_blond, valid_blond_loader, STAGE_1_MODEL, DEVICE)
    df_blond_men['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_blond_men = difficult_examples(stage_1_model, df_blond_men, valid_blond_men_loader, STAGE_1_MODEL, DEVICE)
    df_blond_women['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_blond_women = difficult_examples(stage_1_model, df_blond_women, valid_blond_women_loader, STAGE_1_MODEL, DEVICE)

    df_brunette['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_brunette = difficult_examples(stage_1_model, df_brunette, valid_brunette_loader, STAGE_1_MODEL, DEVICE)
    df_brunette_men['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_brunette_men = difficult_examples(stage_1_model, df_brunette_men, valid_brunette_men_loader, STAGE_1_MODEL, DEVICE)
    df_brunette_women['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_brunette_women = difficult_examples(stage_1_model, df_brunette_women, valid_brunette_women_loader, STAGE_1_MODEL, DEVICE)

df_blond['total_mistakes'] = df_blond.iloc[:,41:].sum(axis=1)
df_blond.to_csv(PATH + "statistics/df_blond.csv", index=False)

df_blond_men['total_mistakes'] = df_blond_men.iloc[:,41:].sum(axis=1)
df_blond_men.to_csv(PATH + "statistics/df_blond_men.csv", index=False)

df_blond_women['total_mistakes'] = df_blond_women.iloc[:,41:].sum(axis=1)
df_blond_women.to_csv(PATH + "statistics/df_blond_women.csv", index=False)

df_brunette['total_mistakes'] = df_brunette.iloc[:,41:].sum(axis=1)
df_brunette.to_csv(PATH + "statistics/df_brunette.csv", index=False)

df_brunette_men['total_mistakes'] = df_brunette_men.iloc[:,41:].sum(axis=1)
df_brunette_men.to_csv(PATH + "statistics/df_brunette_men.csv", index=False)

df_brunette_women['total_mistakes'] = df_brunette_women.iloc[:,41:].sum(axis=1)
df_brunette_women.to_csv(PATH + "statistics/df_brunette_women.csv", index=False)

