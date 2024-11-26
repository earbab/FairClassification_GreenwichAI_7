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

import torchvision.models as models

from arguments import args
       
from jtt_data import train_loader

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
PATH = 'saved_models/run_%s/bs_%s_epochs_%s_lr_%s_wd_%s/'%(RUN, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY)
print(PATH)

torch.manual_seed(RUN)
torch.cuda.manual_seed_all(RUN)
np.random.seed(RUN)
os.environ['PYTHONHASHSEED'] = str(RUN)

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
    if not os.path.exists(PATH+'epoch_%s'%(epoch+1)):
        os.makedirs(PATH+'epoch_%s'%(epoch+1))    
    model.train()
    for batch_idx, (features, _, attributes, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE).unsqueeze(1).float()
        optimizer.zero_grad()
        logits = model(features)
        probas = torch.sigmoid(logits)
        cost = criterion(probas, targets)
        cost.backward()
        optimizer.step()
    CHECKPOINT = PATH + 'epoch_%s/epoch_%s.pt'%(epoch+1, epoch+1)
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, CHECKPOINT)

