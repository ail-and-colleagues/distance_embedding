import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


from test_dataset import Random_Dataset, draw_map

def calc_dist_mtx(x):
    return pairwise_distances(x, x, metric='euclidean', n_jobs=-1)

class TSNE_Embedder():
    def __init__(self, perplexity=30, n_iter=250, s_random=0):
        # self.n_dim = n_dim
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.s_random = s_random

    def fit(self, x):
        # print("TSNE_Embedder x.shape: ", x.shape)
        # kwargs = {
        #     "n_components": 2,
        #     "perplexity": self.perplexity,
        #     "n_iter": self.n_iter,
        #     "random_state": self.s_random,
        #     "init": "pca",
        #     "learning_rate": "auto",
        # }

        print("TSNE_Embedder x.shape: ", x.shape)
        kwargs = {
            "n_components": 2,
            "perplexity": self.perplexity,
            "n_iter": self.n_iter,
            "random_state": self.s_random,
            "init": "random",
            "learning_rate": "auto",
            "metric": "precomputed",
        }
        distance_matrix = calc_dist_mtx(x)
        print("distance_matrix.shape: ", distance_matrix.shape)
        print(distance_matrix[:5, :5])
        
        tsne = TSNE(**kwargs)
        embedded = tsne.fit_transform(distance_matrix)
        return embedded
    

if __name__ == "__main__":

    node_num = 10
    embed_dim = 2
    batch_size = 64
    batch_p_ep = 1000

    train_set = Random_Dataset(node_num, embed_dim, batch_size, batch_p_ep)

    draw_map('./outputs/t-sne-gt.png', train_set.node_pos, range(train_set.node_num))


    tsne = TSNE_Embedder(perplexity=5, n_iter=250000, s_random=0)
    embedded = tsne.fit(train_set.node_pos)
    print(embedded.shape, embedded)

    draw_map('./outputs/t-sne-pred.png', embedded, range(train_set.node_num))

