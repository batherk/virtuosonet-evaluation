import pandas as pd
from utils.paths import get_root_folder

def save_data(df:pd.DataFrame, name='styles'):
    df.to_pickle(f"{get_root_folder()}/data/{name}.pickle")

def load_data(name='styles'):
    return pd.read_pickle(f"{get_root_folder()}/data/{name}.pickle")