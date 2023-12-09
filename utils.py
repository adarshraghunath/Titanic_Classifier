import numpy as np
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import os

class Transforms(): #what is love? baby dont hurt me
    def __init__(self, categorical_features : list[str], numerical_features : list[str]) -> None:
        self.cat = categorical_features
        self.num = numerical_features
        self.imputer = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent') 
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore',min_frequency=5)

    
    def __call__(self, file : pd.DataFrame):
        file[self.num] = self.imputer.fit_transform(file[self.num])
        file[self.num] = self.scaler.fit_transform(file[self.num])
        file[self.cat] = self.imputer_cat.fit_transform(file[self.cat])
        encoded = self.encoder.fit_transform(file[self.cat])
        col_names = self.encoder.get_feature_names_out(self.cat)
        encoded = pd.DataFrame(encoded.toarray(),columns=col_names)
        features = pd.concat([file, encoded], axis=1).drop(self.cat, axis=1)
        
        return features

class CustomDataset(Dataset): #no mo
    def __init__(self,features : pd.DataFrame ,transform = None, training= True, **kwargs) -> None:  #labels : pd.Series
        super().__init__()
        self.features = features
        if training:
            self.labels = kwargs.get('labels',None)
        self.transform = transform
        self.training = training
    
    def __getitem__(self, index):
        if self.training:
            sample = {'features': torch.tensor(self.features.values[index],dtype=torch.float32),'labels': torch.tensor(self.labels.iloc[index],dtype=torch.float32)}
        else:
            sample = {'features': torch.tensor(self.features.values[index],dtype=torch.float32)}
        
        return sample

    def __len__(self):
        return len(self.features)

title_dict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Royalty",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

def save_to_csv(test_file : pd.DataFrame, predictions : np.ndarray , **kwargs) -> None:
    
    dir = os.getcwd()
    pass_id = test_file['PassengerId'].values
    pass_id = pass_id.reshape(-1,1)
    
    fin = np.concatenate([pass_id,predictions],axis=1)
    fin = pd.DataFrame(fin,columns=['PassengerId','Survived'])

    fin.to_csv(path_or_buf=f'{dir}/submissions/submission_{kwargs.get("expt")}.csv',index=False)

def exec_name(data : pd.DataFrame) -> pd.DataFrame:
    
    extract = lambda n : n.split(',')[1].split('.')[0].strip()

    data['Title']  = data['Name'].map(extract)

    data['Title']  = data.Title.map(title_dict)

    data.drop(columns='Name',inplace=True)

    return data

