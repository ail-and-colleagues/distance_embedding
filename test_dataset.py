import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt


class Random_Dataset(Dataset):
    def __init__(self, node_num, pos_dim, batch_size, batch_p_ep):
        self.node_num = node_num
        self.pos_dim = pos_dim
        self.batch_size = batch_size
        self.batch_p_ep = batch_p_ep
        
        # This dataset aims to test the feasibility of estimating nodes' position from distances (in actual uses, represented by not Euclid distances).
        # Thus the dataset provides a pair of nodes as input(x) and the distance between the pair as the ground truth (y).
        self.node_pos = np.random.rand(self.node_num, self.pos_dim)

    def __len__(self):
        # __len__ returns the number of data to be input in one epoch.
        return self.batch_size * self.batch_p_ep
    
    def __getitem__(self, index):
        # x (inputs of a network)
        a, b = np.random.choice(self.node_num, 2, replace=False)
        pos_a = self.node_pos[a]
        pos_b = self.node_pos[b]
        # y (grandtruth)
        dist = np.linalg.norm(pos_b - pos_a)
        return (a.astype(np.int64), b.astype(np.int64)), dist.astype(np.float32)

def draw_map(dst, data, labels):
    
    x = data[:, 0]
    y = data[:, 1]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(x, y, s=50)
    for i in range(len(labels)):
        l = labels[i]
        ax1.annotate(l,xy=(x[i],y[i]),size=15,color="red")
    # plt.show()
    plt.axis('equal')
    plt.savefig(dst)
    plt.close()

if __name__ == "__main__":
    n = 10
    embed_dim = 2
    batch_size = 32
    batch_p_ep = 1000

    train_set = Random_Dataset(n, embed_dim, batch_size, batch_p_ep)

    (a, b), dist = train_set.__getitem__(0)
    print('a: ', a.shape)
    print(a)
    print('b: ', b.shape)
    print(b)
    print('dist: ', dist.shape)
    print(dist)

    draw_map(train_set.node_pos, range(train_set.node_num))
    
        