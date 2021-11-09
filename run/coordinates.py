from utils.data_handling import load_data
import numpy as np


def scalar_projection(x,y):
    return np.dot(x, y) / np.linalg.norm(y)

def vector_projection(x,y):
    return y * np.dot(x, y) / np.dot(y, y)

data_df = load_data('all_styles_100')
dimension_df = load_data('disentangled_dimensions')
mask = (data_df['style_name'] == 'Sad') | (data_df['style_name'] == 'Enjoy')

data = data_df[mask].loc[:,'l0':].to_numpy()
vectors = dimension_df.loc[:,'l0':].to_numpy()

delta = vector_projection(data, vectors)
print(delta)



