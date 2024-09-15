import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch
from ReadData.DatePreProcess import read_df, user_df_preprocess, item_df_preprocess


def interval_df(user_df, item_df):
    df = pd.merge(user_df, item_df, on=['item_id'], how='left')
    # 特征处理及归一化
    return df






if __name__ == '__main__':
    user_df = read_df('../tianchi_fresh_comp_train_user_online.csv', type='user')
    item_df = read_df('../tianchi_fresh_comp_train_item_online.csv', type='item')
    print(user_df.columns)
    print(item_df)
    df = interval_df(user_df, item_df)
    print(df.head())
