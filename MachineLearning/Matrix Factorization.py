import pandas as pd
import numpy as np
from tqdm import tqdm

from ReadData.DatePreProcess import read_df, user_df_preprocess, item_df_preprocess


def init_matrix(user_df, item_df, embedding_dim=64):
    user_idx_list = user_df['user_id'].unique().tolist()
    item_idx_list = item_df['item_id'].unique().tolist()

    user_number, item_number = len(user_idx_list), len(item_idx_list)
    user_matrix = np.random.rand(user_number, embedding_dim)
    item_matrix = np.random.rand(embedding_dim, item_number)

    tar_df = pd.merge(user_df, item_df, how='inner', on='item_id')

    tar_matrix = np.zeros((user_number, item_number), dtype=int)

    repurchase_df = tar_df.loc[tar_df['behavior_type'] >= 2][['user_id', 'item_id']]

    for idx, rows in repurchase_df.iterrows():
        user_id, item_id = rows['user_id'], rows['item_id']
        user_idx = user_idx_list.index(user_id)
        item_idx = item_idx_list.index(item_id)

        tar_matrix[user_idx][item_idx] = 1

    return user_matrix, item_matrix, tar_matrix


def loss_func(user_matrix, item_matrix, tar_matrix):
    # 使用矩阵运算计算损失
    diff = tar_matrix - np.dot(user_matrix, item_matrix.T)  # 注意转置item_matrix
    loss = np.sum(diff ** 2)

    return loss


def update_factors(user_matrix, item_matrix, gradients_user, gradients_item, lr, Lambda):
    # 更新用户和物品矩阵
    user_matrix -= lr * (gradients_user + Lambda * user_matrix)
    item_matrix -= lr * (gradients_item + Lambda * item_matrix)
    return user_matrix, item_matrix


def matrix_factorization(user_matrix, item_matrix, tar_matrix, epochs, lr, Lambda=0.1):
    for epoch in tqdm(range(epochs)):
        # 计算预测矩阵
        pred_matrix = np.dot(user_matrix, item_matrix)

        # 计算梯度
        error_matrix = tar_matrix - pred_matrix
        gradients_user = -2 * np.dot(error_matrix, item_matrix.T)
        gradients_item = -2 * np.dot(user_matrix.T, error_matrix)

        # 添加正则化项
        gradients_user += 2 * Lambda * user_matrix
        gradients_item += 2 * Lambda * item_matrix
        # 更新因子矩阵
        update_factors(user_matrix, item_matrix, gradients_user, gradients_item, lr, Lambda)

        # 计算并打印损失
        loss = np.sum(error_matrix ** 2)
        print('=' * 20 + f'epoch:{epoch}, loss:{loss}' + '=' * 20)


if __name__ == '__main__':
    user_df = read_df('../tianchi_fresh_comp_train_user_online.csv', type='user')
    item_df = read_df('../tianchi_fresh_comp_train_item_online.csv', type='item')
    embedding_dim = 16
    epochs = 90
    lr = 1e-4
    Lambda = 0.01
    user_matrix, item_matrix, tar_matrix = init_matrix(user_df, item_df, embedding_dim)
    matrix_factorization(user_matrix, item_matrix, tar_matrix, epochs=epochs, lr=lr, Lambda=Lambda)

