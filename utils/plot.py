from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_latents(styles_df, dimensions=3):
    x = styles_df.loc[:,'qpm':]

    labels = list(styles_df['style_name'])
    unique_labels = list(set(labels))
    classes = [unique_labels.index(label) for label in labels]

    tsne = TSNE(n_components=dimensions).fit_transform(X=x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(*zip(*tsne), c=classes)
    points, classes = scatter.legend_elements()
    class_legend = ax.legend(points, labels, loc="lower left", title="Classes")
    ax.add_artist(class_legend)
    plt.show()