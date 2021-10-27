#%%
from sklearn import svm
from utils.data_handling import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

styles = load_data('multiple_styles')
anger = styles[styles['style_name'] =='Anger']
sad = styles[styles['style_name'] == 'Sad']
anger_x = anger.loc[:,'l0':]
sad_x = sad.loc[:,'l0':]
labels = ['Anger', 'Sad']

X = np.append(anger_x.to_numpy(), sad_x.to_numpy(), axis=0)
y = np.append(np.zeros(len(anger_x)), np.ones(len(sad_x)))

classifier = svm.SVC(kernel='linear')
classifier.fit(X, y)

pca = PCA(n_components=3)
coords = pca.fit_transform(X=X)

classes = [int(i)for i in y]
fig = plt.figure()

xs, ys, zs = zip(*coords)
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)
min_z, max_z = min(zs), max(zs)
delta = 0.2

grid_x = np.arange(min_x, max_x, delta)
grid_y = np.arange(min_y, max_y, delta)
grid_z = np.arange(min_z, max_z, delta)
xx2d, yy2d = np.meshgrid(grid_x, grid_y)

xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z)
latent_grid = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
predictions = classifier.predict(latent_grid).reshape(yy.shape)
Z = np.array([[grid_z[np.where(row == 0.)[0][0]] if 0. in row else max_z for row in col ] for col in predictions])

ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx2d, yy2d, Z)
scatter = ax.scatter(xs, ys, zs, c=classes)
points, classes = scatter.legend_elements()
class_legend = ax.legend(points, labels, loc="lower left", title="Classes")
ax.add_artist(class_legend)
plt.show()


