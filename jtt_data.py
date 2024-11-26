import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from arguments import args

class CelebA():
  '''Wraps the celebA dataset, allowing an easy way to:
       - Select the features of interest,
       - Split the dataset into 'training', 'test' or 'validation' partition.
  '''
  def __init__(self, main_folder='./celeba-dataset/', selected_features=None, drop_features=[]):
    self.main_folder = main_folder
    self.images_folder   = os.path.join(main_folder, 'img_align_celeba/')
    self.attributes_path = os.path.join(main_folder, 'list_attr_celeba.csv')
    self.partition_path  = os.path.join(main_folder, 'list_eval_partition.csv')
    self.selected_features = selected_features
    self.features_name = []
    self.__prepare(drop_features)

  def __prepare(self, drop_features):
    '''do some preprocessing before using the data: e.g. feature selection'''
    # attributes:
    if self.selected_features is None:
      self.attributes = pd.read_csv(self.attributes_path)
      self.num_features = 40
    else:
      self.num_features = len(self.selected_features)
      self.selected_features = self.selected_features.copy()
      self.selected_features.append('image_id')
      self.attributes = pd.read_csv(self.attributes_path)[self.selected_features]

    # remove unwanted features:
    for feature in drop_features:
      if feature in self.attributes:
        self.attributes = self.attributes.drop(feature, axis=1)
        self.num_features -= 1
      
    self.attributes.set_index('image_id', inplace=True)
    self.attributes.replace(to_replace=-1, value=0, inplace=True)
    self.attributes['image_id'] = list(self.attributes.index)
  
    self.features_name = list(self.attributes.columns)[:-1]
  
    # load ideal partitioning:
    self.partition = pd.read_csv(self.partition_path)
    self.partition.set_index('image_id', inplace=True)
  
  def split(self, name='training', drop_zero=False):
    '''Returns the ['training', 'validation', 'test'] split of the dataset'''
    # select partition split:
    if name is 'training':
      to_drop = self.partition.where(lambda x: x != 0).dropna()
    elif name is 'validation':
      to_drop = self.partition.where(lambda x: x != 1).dropna()
    elif name is 'test':  # test
      to_drop = self.partition.where(lambda x: x != 2).dropna()
    else:
      raise ValueError('CelebA.split() => `name` must be one of [training, validation, test]')

    partition = self.partition.drop(index=to_drop.index)
      
    # join attributes with selected partition:
    joint = partition.join(self.attributes, how='inner').drop('partition', axis=1)

    if drop_zero is True:
      # select rows with all zeros values
      return joint.loc[(joint[self.features_name] == 1).any(axis=1)]
    elif 0 <= drop_zero <= 1:
      zero = joint.loc[(joint[self.features_name] == 0).all(axis=1)]
      zero = zero.sample(frac=drop_zero)
      return joint.drop(index=zero.index)

    return joint

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, df, img_dir, transform=None):
    
        self.img_dir = img_dir
        self.img_names = df['image_id'].values
        self.y = df['Blond_Hair'].values
        self.attribute = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)

        name = self.img_names[index]
        attribute = self.attribute[index]
        label = self.y[index]
        
        return img, name, attribute, label

    def __len__(self):
        return self.y.shape[0]

BATCH_SIZE = args.batch_size

celeba = CelebA()
train_split = celeba.split('training'  , drop_zero=False)
valid_split = celeba.split('validation', drop_zero=False)
test_split = celeba.split('test', drop_zero=False)

custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       #transforms.Grayscale(),                                       
                                       #transforms.Lambda(lambda x: x/255.),
                                       transforms.ToTensor()])


train_men_mask = train_split['Male'].apply(lambda x: x == 1)
train_women_mask = train_split['Male'].apply(lambda x: x == 0)
train_blond_mask = train_split['Blond_Hair'] == 1
train_brunette_mask = train_split['Blond_Hair'] == 0

train_men = train_split[train_men_mask]
train_women = train_split[train_women_mask]
train_blond_men = train_split[train_blond_mask & train_men_mask]
train_brunette_men = train_split[train_men_mask & train_brunette_mask]
train_blond_women = train_split[train_blond_mask & train_women_mask]
train_brunette_women = train_split[train_women_mask & train_brunette_mask]
train_blond = train_split[train_blond_mask]
train_brunette = train_split[train_brunette_mask]

train_dataset = CelebaDataset(df=train_split, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
train_men_dataset = CelebaDataset(df=train_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
train_women_dataset = CelebaDataset(df=train_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
train_blond_men_dataset = CelebaDataset(df=train_blond_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
train_brunette_men_dataset = CelebaDataset(df=train_brunette_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
train_blond_women_dataset = CelebaDataset(df=train_blond_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
train_brunette_women_dataset = CelebaDataset(df=train_brunette_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
train_blond_dataset = CelebaDataset(df=train_blond, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
train_brunette_dataset = CelebaDataset(df=train_brunette, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)


train_loader = DataLoader(dataset=train_dataset,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  num_workers=4, 
                  pin_memory=True)      
train_blond_men_loader = DataLoader(dataset=train_blond_men_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
train_blond_women_loader = DataLoader(dataset=train_blond_women_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=4)
train_brunette_men_loader = DataLoader(dataset=train_brunette_men_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=4)
train_brunette_women_loader = DataLoader(dataset=train_brunette_women_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=4)                     
train_blond_loader = DataLoader(dataset=train_blond_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 num_workers=4, 
                 pin_memory=True)
train_brunette_loader = DataLoader(dataset=train_brunette_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 num_workers=4, 
                 pin_memory=True)     

print("Shape of train dataset: ", train_split.shape)
print("Shape of train dataset with Blond Hair", train_blond.shape)
print("Shape of train dataset with non-Blond Hair", train_brunette.shape)
print("Shape of train dataset with Men and Blond Hair: ", train_blond_men.shape)
print("Shape of train dataset with Women and Blond Hair: ", train_blond_women.shape)
print("Shape of train dataset with Men and non-Blond Hair: ", train_brunette_men.shape)
print("Shape of train dataset with Women and non-Blond Hair: ", train_brunette_women.shape)
print("Shape of train dataset with Men: ", train_men.shape)
print("Shape of train dataset with Women: ", train_women.shape)

# Unbiased training data
unbiased_train_split  = pd.DataFrame()   
unbiased_train_brunette = train_brunette.sample(n=24267, random_state=args.run)
# unbiased_train_split = train_blond.append(unbiased_train_brunette, ignore_index=True)
unbiased_train_split = pd.concat([train_blond, unbiased_train_brunette], ignore_index=True)
unbiased_train_dataset = CelebaDataset(df=unbiased_train_split, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
unbiased_train_brunette_dataset = CelebaDataset(df=unbiased_train_brunette, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
unbiased_train_loader = DataLoader(dataset=unbiased_train_dataset,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  num_workers=4, 
                  pin_memory=True)
unbiased_train_brunette_loader = DataLoader(dataset=unbiased_train_brunette_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 num_workers=4, 
                 pin_memory=True)  
print("Shape of unbiased train dataset: ", unbiased_train_split.shape)
print("Shape of unbiased train brunette dataset: ", unbiased_train_brunette.shape)

valid_men_mask = valid_split['Male'].apply(lambda x: x == 1)
valid_women_mask = valid_split['Male'].apply(lambda x: x == 0)
valid_blond_mask = valid_split['Blond_Hair'] == 1
valid_brunette_mask = valid_split['Blond_Hair'] == 0

valid_men = valid_split[valid_men_mask]
valid_women = valid_split[valid_women_mask]
valid_blond_men = valid_split[valid_blond_mask & valid_men_mask]
valid_brunette_men = valid_split[valid_men_mask & valid_brunette_mask]
valid_blond_women = valid_split[valid_blond_mask & valid_women_mask]
valid_brunette_women = valid_split[valid_women_mask & valid_brunette_mask]
valid_blond = valid_split[valid_blond_mask]
valid_brunette = valid_split[valid_brunette_mask]

valid_dataset = CelebaDataset(df=valid_split, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
valid_men_dataset = CelebaDataset(df=valid_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
valid_women_dataset = CelebaDataset(df=valid_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
valid_blond_men_dataset = CelebaDataset(df=valid_blond_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
valid_brunette_men_dataset = CelebaDataset(df=valid_brunette_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
valid_blond_women_dataset = CelebaDataset(df=valid_blond_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
valid_brunette_women_dataset = CelebaDataset(df=valid_brunette_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
valid_blond_dataset = CelebaDataset(df=valid_blond, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
valid_brunette_dataset = CelebaDataset(df=valid_brunette, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)

valid_loader = DataLoader(dataset=valid_dataset,
                 batch_size=1024,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)
valid_men_loader = DataLoader(dataset=valid_men_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
valid_women_loader = DataLoader(dataset=valid_women_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
valid_blond_men_loader = DataLoader(dataset=valid_blond_men_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
valid_blond_women_loader = DataLoader(dataset=valid_blond_women_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
valid_brunette_men_loader = DataLoader(dataset=valid_brunette_men_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
valid_brunette_women_loader = DataLoader(dataset=valid_brunette_women_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
valid_blond_loader = DataLoader(dataset=valid_blond_dataset,
                 batch_size=1024,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)
valid_brunette_loader = DataLoader(dataset=valid_brunette_dataset,
                 batch_size=1024,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)

print("Shape of valid dataset: ", valid_split.shape)
print("Shape of valid dataset with Blond Hair", valid_blond.shape)
print("Shape of valid dataset with non-Blond Hair", valid_brunette.shape)
print("Shape of valid dataset with Men and Blond Hair: ", valid_blond_men.shape)
print("Shape of valid dataset with Women and Blond Hair: ", valid_blond_women.shape)
print("Shape of valid dataset with Men and non-Blond Hair: ", valid_brunette_men.shape)
print("Shape of valid dataset with Women and non-Blond Hair: ", valid_brunette_women.shape)
print("Shape of valid dataset with Men: ", valid_men.shape)
print("Shape of valid dataset with Women: ", valid_women.shape)

test_men_mask = test_split['Male'].apply(lambda x: x == 1)
test_women_mask = test_split['Male'].apply(lambda x: x == 0)
test_blond_mask = test_split['Blond_Hair'] == 1
test_brunette_mask = test_split['Blond_Hair'] == 0

test_men = test_split[test_men_mask]
test_women = test_split[test_women_mask]
test_blond_men = test_split[test_blond_mask & test_men_mask]
test_brunette_men = test_split[test_men_mask & test_brunette_mask]
test_blond_women = test_split[test_blond_mask & test_women_mask]
test_brunette_women = test_split[test_women_mask & test_brunette_mask]
test_blond = test_split[test_blond_mask]
test_brunette = test_split[test_brunette_mask]

test_dataset = CelebaDataset(df=test_split, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
test_men_dataset = CelebaDataset(df=test_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
test_women_dataset = CelebaDataset(df=test_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
test_blond_men_dataset = CelebaDataset(df=test_blond_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
test_brunette_men_dataset = CelebaDataset(df=test_brunette_men, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
test_blond_women_dataset = CelebaDataset(df=test_blond_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
test_brunette_women_dataset = CelebaDataset(df=test_brunette_women, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
test_blond_dataset = CelebaDataset(df=test_blond, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)
test_brunette_dataset = CelebaDataset(df=test_brunette, img_dir='./celeba-dataset/img_align_celeba/img_align_celeba/', transform=custom_transform)

test_loader = DataLoader(dataset=test_dataset,
                 batch_size=1024,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)
test_men_loader = DataLoader(dataset=test_men_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
test_women_loader = DataLoader(dataset=test_women_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
test_blond_men_loader = DataLoader(dataset=test_blond_men_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
test_blond_women_loader = DataLoader(dataset=test_blond_women_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
test_brunette_men_loader = DataLoader(dataset=test_brunette_men_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
test_brunette_women_loader = DataLoader(dataset=test_brunette_women_dataset,
                     batch_size=1024,
                     shuffle=False,
                     num_workers=4)
test_blond_loader = DataLoader(dataset=test_blond_dataset,
                 batch_size=1024,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)
test_brunette_loader = DataLoader(dataset=test_brunette_dataset,
                 batch_size=1024,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)

print("Shape of test dataset: ", test_split.shape)
print("Shape of test dataset with Blond Hair", test_blond.shape)
print("Shape of test dataset with non-Blond Hair", test_brunette.shape)
print("Shape of test dataset with Men and Blond Hair: ", test_blond_men.shape)
print("Shape of test dataset with Women and Blond Hair: ", test_blond_women.shape)
print("Shape of test dataset with Men and non-Blond Hair: ", test_brunette_men.shape)
print("Shape of test dataset with Women and non-Blond Hair: ", test_brunette_women.shape)
print("Shape of test dataset with Men: ", test_men.shape)
print("Shape of test dataset with Women: ", test_women.shape)

