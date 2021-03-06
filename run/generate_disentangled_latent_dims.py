import numpy as np
from utils.data_handling import load_data, save_data
from sklearn import svm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import itertools

SAVE_DATA = True
PLOT_ROC = True
MASK = False
MASK_NAME = 'composer_and_piece'
SAVE_NAME = f"disentangled_dimensions"

if MASK :
    SAVE_NAME += f"_mask_{MASK_NAME}"



df = load_data('styles')

styles = df['style_name'].unique()
mask = (df['composer'] == 'Bach') & (df['piece'] == "french-suite_bwv812_no1_allemande")

if MASK:
    df = df[mask]

dimensions = itertools.combinations(styles, 2)

disentangled_dimensions = []

for dimension_start, dimension_end in dimensions:
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

    if PLOT_ROC:
        classified_dec = classifier.decision_function(X)
        xs, ys, _ = roc_curve(y, classified_dec)
        plt.plot(xs, ys, label=f"{dimension_start} - {dimension_end}")
        plt.legend()
        plt.show()

    for i, value in enumerate(dimension_vector):
        disentangled_dimension[f"l{i}"] = value

    disentangled_dimensions.append(disentangled_dimension)

disentangled_dimensions_df = pd.DataFrame(disentangled_dimensions)
print(disentangled_dimensions_df)
if SAVE_DATA:
    save_data(disentangled_dimensions_df, SAVE_NAME)