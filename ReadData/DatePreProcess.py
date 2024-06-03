import pandas as pd


def item_df_preprocess(df):
    df = df.head(10000)
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
