import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib  # <--ここを追加
matplotlib.use('Agg')  # <--ここを追加
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt

import datetime
import convert_dataset

class Random_Dataset(Dataset):
    def __init__(self, node_num_sum, pos_dim, batch_size, batch_p_ep,dis_mtx,node_pos):
        self.node_num_sum = node_num_sum    #N,駅の数
        self.pos_dim = pos_dim      #d,平面図上ならpos_dim=2
        self.batch_size = batch_size
        self.batch_p_ep = batch_p_ep
        self.dis_mtx = dis_mtx
        
        # This dataset aims to test the feasibility of estimating nodes' position from distances (in actual uses, represented by not Euclid distances).
        # Thus the dataset provides a pair of nodes as input(x) and the distance between the pair as the ground truth (y).
        
        #各点の座標を定義:convert_dataset.pyにて記述。
        self.node_pos=node_pos

    def __len__(self):
        # __len__ returns the number of data to be input in one epoch.
        return self.batch_size * self.batch_p_ep
    
    def __getitem__(self, index):
        #多分ランダムな組み合わせで距離を生成してる
        
        # x (inputs of a network)
        a, b = np.random.choice(self.node_num_sum, 2, replace=False)
        dist_matrix=self.dis_mtx
        dist = dist_matrix[a,b]
        return (a.astype(np.int64), b.astype(np.int64)), dist.astype(np.float32)
        

def draw_map(dst, data, labels, node_color):
    
    x = data[:, 0]
    y = data[:, 1]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    for i ,px,py,pc in zip(range(len(labels)), x, y, node_color):
        l = labels[i]
        ax1.scatter(px, py, s=50, c=pc)
        ax1.annotate(l,xy=(x[i],y[i]),size=11,color="red")
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
    #print('b_0:', b,b.shape)
    #print('dist_0:', dist,dist.shape)
    print(type(dist))        
    
    #---------test--------------
    dt_set = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    #---------------------------

    draw_map(dt_set,train_set.node_pos, range(train_set.node_num))
    
        