import numpy as np
from utils.data_handling import load_data, save_data
from sklearn import svm
import pandas as pd

SAVE_DATA = False
SAVE_NAME = 'disentangled_dimensions'
DIMENSIONS = [['Sad', 'Enjoy'], ['Relax', 'Anger']]

df = load_data('all_styles_100')

disentangled_dimensions = []

for dimension_start, dimension_end in DIMENSIONS:
    start_samples = df[df['style_name'] == dimension_start].loc[:,'l0':].to_numpy()
    end_samples = df[df['style_name'] == dimension_end].loc[:,'l0':].to_numpy()

    X = np.append(start_samples, end_samples, axis=0)
    y = np.append(np.zeros(len(start_samples)), np.ones(len(end_samples)))

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X, y)

    disentangled_dimension = {
        'negative_name': dimension_start,
        'positive_name': dimension_end,
        'direction_intercept': classifier.intercept_
    }

    dimension_vector = classifier.coef_.squeeze()

    for i, value in enumerate(dimension_vector):
        disentangled_dimension[f"l{i}"] = value

    disentangled_dimensions.append(disentangled_dimension)

disentangled_dimensions_df = pd.DataFrame(disentangled_dimensions)
print(disentangled_dimensions_df)
if SAVE_DATA:
    save_data(disentangled_dimensions_df, SAVE_NAME)