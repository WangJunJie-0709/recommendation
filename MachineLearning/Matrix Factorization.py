import pandas as pd
import numpy as np

from ReadData.DatePreProcess import read_df, user_df_preprocess, item_df_preprocess

user_src_name = '../tianchi_fresh_comp_train_user_online.csv'
user_df = read_df(user_src_name, type='user')
print(user_df)