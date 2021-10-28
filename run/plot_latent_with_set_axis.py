#%%
from sklearn import svm
from utils.data_handling import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.matrix import get_dimensionality_reduction_matrix
from utils.data_handling import save_data
import pandas as pd

SAVE_DATA = True

styles = load_data('multiple_styles')
anger = styles[styles['style_name'] =='Anger']
sad = styles[styles['style_name'] == 'Sad']
anger_x = anger.loc[:,'l0':].to_numpy()
sad_x = sad.loc[:,'l0':].to_numpy()
labels = ['Anger', 'Sad']

X = np.append(anger_x, sad_x, axis=0)
y = np.append(np.zeros(len(anger_x)), np.ones(len(sad_x)))

classifier = svm.SVC(kernel='linear')
classifier.fit(X, y)

classification_vector = classifier.coef_.squeeze()
xs = X.dot(classification_vector).ravel()
reduction_matrix = get_dimensionality_reduction_matrix(classification_vector)
X_transformed = X.dot(reduction_matrix)

pca = PCA(n_components=2)
coords = pca.fit_transform(X=X_transformed)

fig = plt.figure()

ys, zs = zip(*coords)
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)
min_z, max_z = min(zs), max(zs)
delta = 0.2
yy, zz = np.meshgrid(np.arange(min_y, max_y, delta), np.arange(min_z, max_z, delta))
xx = np.ones(yy.shape)*classifier.intercept_

if SAVE_DATA:
    classification_data = {
        'x': xs,
        'y': ys,
        'z': zs,
        'label': y
    }
    df = pd.DataFrame(classification_data)
    save_data(df, 'latent_classification')

anger_mean = [np.mean(values[:len(anger_x)]) for values in [xs, ys, zs]]
sad_mean = [np.mean(values[len(anger_x):]) for values in [xs, ys, zs]]

ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(xs, ys, zs, c=y)
surf = ax.plot_surface(xx, yy, zz, alpha=0.5)
anger_mean_scatter = ax.scatter(*anger_mean, c='blue', s=100)
sad_mean_scatter = ax.scatter(*sad_mean, c='orange', s=100)
legends = scatter.legend_elements()[0]
ax.set_xlabel(f"Anger mean: {anger_mean[0]:.2f}, Sad mean {sad_mean[0]:.2f}")
class_legend = ax.legend(legends, labels, loc="lower left", title="Classes")
ax.add_artist(class_legend)
plt.show()


