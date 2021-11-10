from utils.data_handling import load_data
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


def scalar_projection(x,y):
    return np.dot(x, y) / np.linalg.norm(y)

def vector_projection(x,y):
    return y * np.dot(x, y) / np.dot(y, y)


def get_coordinates(latent_vectors, dimension_vectors, output_dimensions):
    scalar_projections = np.array([[scalar_projection(x,y) for x in latent_vectors] for y in dimension_vectors]).transpose()
    if output_dimensions <= dimension_vectors.shape[0]:
        return scalar_projections[:,:output_dimensions]
    else:
        remaining_dimensions = output_dimensions - dimension_vectors.shape[0]
        pca = PCA(n_components=remaining_dimensions)
        pca_coords = pca.fit_transform(X=latent_vectors)
        return np.append(scalar_projections, pca_coords, axis=1)

data_df = load_data('all_styles_100')
dimension_df = load_data('disentangled_dimensions')
mask = (data_df['style_name'] == 'Sad') | (data_df['style_name'] == 'Enjoy')

latent_vectors = data_df[mask].loc[:,'l0':].to_numpy()
dimension_vectors = dimension_df.loc[:,'l0':].to_numpy()

style_name = data_df[mask]['style_name'].reset_index(drop=True)
labels = style_name.apply(lambda x: 1 if x == 'Enjoy' else 0)
coordinates = get_coordinates(latent_vectors, dimension_vectors, 3)

all = pd.concat([style_name, labels, pd.DataFrame(coordinates, columns=['x', 'y', 'z'])], axis=1)


