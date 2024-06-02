import pandas as pd


def item_df_preprocess(df):

    return df


def user_df_preprocess(df):
    df[['date', 'hr']] = df['time'].str.split(' ', expand=True)
    df = df.drop(columns='time')
    return df

def read_df(src_name, type=None):
    df = pd.read_csv(src_name)

    if type == 'user':
        user_df_preprocess(df)
    elif type == 'item':
        item_df_preprocess(df)
    return df



if __name__ == '__main__':
    item_src_name = '../tianchi_fresh_comp_train_item_online.csv'
    user_src_name = '../tianchi_fresh_comp_train_user_online.csv'
    item_df = read_df(item_src_name, type='item')
    user_df = read_df(user_src_name, type='user')
    print(user_df.head(5))
