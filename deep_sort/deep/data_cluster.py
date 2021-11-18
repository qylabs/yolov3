import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets


# '''t-SNE'''
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
# X_tsne = tsne.fit_transform(X)

# print("Org data is {}. Embedded data is {}".format(X.shape, X_tsne.shape))
 
# '''嵌入空间可视化'''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# # plt.show()
# plt.savefig('tsne.jpg')


def scatter(x, labels,n_class):
    #X is 2D ndarray
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,c=plt.cm.Set1(labels))
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
 
    # We add the labels for each digit.
    txts = []
    for i in range(n_class):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txts.append(txt)

    return f, ax, sc, txts


def TSNE_cluster(X,y,n_class,plot_save):
    digits_proj = manifold.TSNE(n_components=2, init='random', random_state=501).fit_transform(X)
    scatter(digits_proj, y,n_class)
    plt.title('TSNE')
    plt.savefig(plot_save, dpi=120)
    print('TSNE plot saved!')


def MDS_cluster(X,y,n_class,plot_save):
    digits_proj = manifold.MDS(n_components=2).fit_transform(X)
    scatter(digits_proj, y,n_class)
    plt.title('MDS')
    plt.savefig(plot_save, dpi=120)
    print('MDS plot saved!')



if __name__=="__main__":
    digits = datasets.load_digits(n_class=6)
    X, y = digits.data, digits.target
    
    TSNE_cluster(X,y,n_class=6,plot_save='tsne-generated.jpg')
