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


def matrix_factorization(user_matrix, item_matrix, tar_matrix, epoches, lr, Lambda):
    embedding_dims = len(user_matrix[0])
    for epoch in tqdm(range(epoches)):
        for i in range(len(user_matrix)):
            for j in range(len(item_matrix[0])):
                eij = tar_matrix[i, j] - np.dot(user_matrix[i, :], item_matrix[:, j])
                for embedding_dim in range(embedding_dims):
                    user_matrix[i][embedding_dim] += lr * (2 * eij * item_matrix[embedding_dim][j] - Lambda * user_matrix[i][embedding_dim])
                    item_matrix[embedding_dim][j] += lr * (2 * eij * user_matrix[i][embedding_dim] - Lambda * item_matrix[embedding_dim][j])

        loss = loss_func(user_matrix, item_matrix, tar_matrix)
        print('=' * 20 + f'epoch:{epoch}, loss:{loss}' + '=' * 20)


if __name__ == '__main__':
    user_df = read_df('../tianchi_fresh_comp_train_user_online.csv', type='user')
    item_df = read_df('../tianchi_fresh_comp_train_item_online.csv', type='item')
    embedding_dim = 16
    epoches = 30
    lr = 0.001
    Lambda = 0.01
    user_matrix, item_matrix, tar_matrix = init_matrix(user_df, item_df, embedding_dim)
    matrix_factorization(user_matrix, item_matrix, tar_matrix, epoches=epoches, lr=lr, Lambda=Lambda)

