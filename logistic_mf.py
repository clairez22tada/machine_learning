import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["user", "item"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, seed=23):
        super(MF, self).__init__()
        torch.manual_seed(seed)
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        # init 
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_i = self.item_bias(v).squeeze()
        
        dot_product = (U*V).sum(1) +  b_u  + b_i
        y_hat = torch.sigmoid(dot_product)

        return y_hat
    
def train_one_epoch(model, train_df, optimizer):
    """ Trains the model for one epoch"""
    model.train()
    """Trains the model for one epoch"""
    user = torch.LongTensor(train_df['user'].values)
    item = torch.LongTensor(train_df['item'].values)
    rating = torch.FloatTensor(train_df['rating'].values)

    y_hat = model(user, item)
    loss = F.binary_cross_entropy_with_logits(y_hat, rating.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    
    return train_loss

def valid_metrics(model, valid_df):
    """Computes validation loss and accuracy"""
    model.eval()
    user = torch.LongTensor(valid_df['user'].values)
    item = torch.LongTensor(valid_df['item'].values)
    rating = torch.FloatTensor(valid_df['rating'].values)

    with torch.no_grad():
        y_hat = model(user, item)
        valid_loss = F.binary_cross_entropy_with_logits(y_hat, rating.float()).item()
        y_pred = (y_hat>0.5).numpy()
        y_true =rating.numpy()
        valid_acc = accuracy_score(y_true, y_pred)

    return valid_loss, valid_acc

def training(model, train_df, valid_df, epochs=10, lr=0.01, wd=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for i in range(epochs):
        train_loss = train_one_epoch(model, train_df, optimizer)
        valid_loss, valid_acc = valid_metrics(model, valid_df) 
        print("train loss %.3f valid loss %.3f valid acc %.3f" % (train_loss, valid_loss, valid_acc)) 

