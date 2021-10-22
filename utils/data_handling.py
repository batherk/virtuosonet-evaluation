import pandas as pd

def save_styles_df(df:pd.DataFrame, name='styles',path='./../data/'):
    df.to_pickle(f"{path}{name}.pkl")

def load_styles_df(name='styles',path='./../data/'):
    return pd.read_pickle(f"{path}{name}.pkl")