import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

import numpy as np
import pandas as pd
from PIL import Image
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import torchvision.models as models
from torchvision.utils import save_image

from arguments import args
import matplotlib.pyplot as plt

from jtt_data import custom_transform
from jtt_data import CelebaDataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##########################
### SETTINGS
##########################

# Hyperparameters
DEVICE = 'cuda:0'
BASE_LR = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MOMENTUM = args.momentum
WEIGHT_DECAY = args.weight_decay
RUN = args.run
STAGE_1_EPOCHS = [args.stage_1_epoch]
UPSAMPLE = args.upsample
PATH = 'saved_models/run_%s/bs_%s_epochs_%s_lr_%s_wd_%s/'%(RUN, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY)
print(PATH)

torch.manual_seed(RUN)
torch.cuda.manual_seed_all(RUN)
np.random.seed(RUN)
os.environ['PYTHONHASHSEED'] = str(RUN)
    
for STAGE_1_MODEL in STAGE_1_EPOCHS:  
    from jtt_data import train_split, train_blond_loader, train_brunette_loader

    model = models.resnet50(pretrained=True)
    d = model.fc.in_features
    model.fc = nn.Linear(d, 1)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0])
    model.to(DEVICE)    
    load_path = PATH + 'epoch_%s/epoch_%s.pt'%(STAGE_1_MODEL, STAGE_1_MODEL)
    print('Loading Model Checkpoint: ', load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    df_blond  = pd.DataFrame()
    with torch.no_grad():
        for i, (features, names, attributes, targets) in enumerate(train_blond_loader):
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()
            logits = model(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5 
            incorrect_predictions = torch.where(predicted_labels != targets)[0]
            incorrect_predictions = list(incorrect_predictions.to('cpu').numpy())
            names = [names[i] for i in incorrect_predictions]
            df_blond = df_blond.append(train_split.loc[train_split['image_id'].isin(names)], ignore_index=True)
    
    df_brunette  = pd.DataFrame()    
    with torch.no_grad():
        for i, (features, names, attributes, targets) in enumerate(train_brunette_loader):
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()
            logits = model(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5    
            incorrect_predictions = torch.where(predicted_labels != targets)[0]
            incorrect_predictions = list(incorrect_predictions.to('cpu').numpy())
            names = [names[i] for i in incorrect_predictions]
            df_brunette = df_brunette.append(train_split.loc[train_split['image_id'].isin(names)], ignore_index=True)
    
    # Upsampling 
    for i in range(UPSAMPLE):
        train_split = train_split.append(df_blond, ignore_index=True)
        train_split = train_split.append(df_brunette, ignore_index=True)
    train_dataset = CelebaDataset(df=train_split, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
    train_loader = DataLoader(dataset=train_dataset,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  num_workers=4, 
                  pin_memory=True)
                  
    model = models.resnet50(pretrained=True)
    d = model.fc.in_features
    model.fc = nn.Linear(d, 1)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model, device_ids=[0])
    model.to(DEVICE)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        SAVE_PATH = PATH + 'epoch_%s/stage_2/incorrect_upsampled_%s_bs_%s_epochs_%s_lr_%s_wd_%s/epochs/epoch_%s/'%(STAGE_1_MODEL, UPSAMPLE, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY, epoch+1)
        print(SAVE_PATH)
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        model.train()
        for batch_idx, (features, _, _, targets) in enumerate(train_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).unsqueeze(1).float()
            optimizer.zero_grad()
            logits = model(features)
            probas = torch.sigmoid(logits)
            cost = criterion(probas, targets)
            cost.backward()
            optimizer.step()
        
        CHECKPOINT = SAVE_PATH + 'epoch_%s.pt'%(epoch+1)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, CHECKPOINT)
            
