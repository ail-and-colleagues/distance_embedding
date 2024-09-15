import os
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

import torch
from torch.utils.data import Dataset


import cv2
import numpy as np
import matplotlib.pyplot as plt


def erode_image(img, kernel, iterations=1):
    kernel = np.ones(kernel, np.uint8)
    return cv2.erode(img, kernel, iterations)


class Image_Dataset(Dataset):
    def __init__(self, image_path, hundle_nodes_num, batch_size, batch_p_ep):
        self.image_path = image_path
        self.hundle_nodes_num = hundle_nodes_num
        self.batch_size = batch_size
        self.batch_p_ep = batch_p_ep

        dirname = os.path.dirname(self.image_path)
        basename = os.path.basename(self.image_path)
        basename, ext = os.path.splitext(basename)
        print('image: ', dirname, basename, ext)

        self.image = cv2.imread(self.image_path)
        print('image.shape: ', self.image.shape)

        self.cost_matrix_path = os.path.join(dirname, basename + '.npz')
        try:
            print('load from: ', self.cost_matrix_path)
            npz = np.load(self.cost_matrix_path)

        except:
            print('cannot load... create_cost_matrix')
            self.create_cost_matrix()
            npz = np.load(self.cost_matrix_path)

        print('npz.files: ', npz.files)
        self.nodes = npz['nodes']
        self.cost_matrix = npz['cost_matrix']
        self.predecessors = npz['predecessors']

        self.node_num = self.nodes.shape[0]
        print('nodes.shape: ', self.nodes.shape)
        print('cost_matrix.shape: ', self.cost_matrix.shape)
        print('predecessors.shape: ', self.predecessors.shape)

        self.hundle_nodes_idx = np.random.choice(self.node_num, self.hundle_nodes_num, replace=False)
        # self.hundle_nodes_idx = np.arange(self.node_num)
        # self.hundle_nodes_idx = np.sort(self.hundle_nodes_idx)
        self.hundled_nodes = self.nodes[self.hundle_nodes_idx]
        print('self.hundle_nodes_idx[:10]', self.hundle_nodes_idx[:10])
        print('self.hundled_nodes[:10]', self.hundled_nodes[:10])

        self.hundle_nodes_num = self.hundled_nodes.shape[0]


    # def fetch_pos_from_hundled(self, idx):
    #     i = self.hundle_nodes_idx[idx]
    #     return self.nodes[i]

    def create_cost_matrix(self):

        # eruded = erode_image(image, (3, 3))
        int_area = self.image[:, :, 2] != 127
        wall =  self.image[:, :, 2] != 0
        mask = int_area & wall

        mask = np.where(mask)
        nodes = np.concatenate([mask[0].reshape([-1, 1]), mask[1].reshape([-1, 1])], axis=1)

        # chk
        # t = np.zeros_like(self.image)
        # t[mask] = 255
        # cv2.imshow('t', t)
        # cv2.waitKey(0)
    
        print('nodes.shape: ', nodes.shape)
        print('nodes[:5]: ', nodes[:5])

        graph = list()
        for i, p in enumerate(nodes):
            norm = np.linalg.norm(nodes - p, ord=2, axis=1)
            dsts = np.where(norm<=1.0, 1, 0)
            graph.append(dsts)

        graph = np.array(graph)
        graph[np.arange(nodes.shape[0]), np.arange(nodes.shape[0])] = 0
        
        #chk
        print('graph[:10, :10]: ')
        for r in graph[:10]:
            print(r[:10])
        
        csr = csr_matrix(graph)
        
        print('solve shortest_path')
        cost_matrix, predecessors = shortest_path(csr, return_predecessors=True)

        print('save as: ', self.cost_matrix_path)
        np.savez_compressed(self.cost_matrix_path,
                 nodes=nodes,
                 cost_matrix=cost_matrix,
                 predecessors=predecessors,            
        )
        

    def __len__(self):
        # __len__ returns the number of data to be input in one epoch.
        return self.batch_size * self.batch_p_ep
    
    def __getitem__(self, index):
        # x (inputs of a network)
        a, b = np.random.choice(self.hundle_nodes_num, 2, replace=False)

        # pos_a = self.node_pos[a]
        # pos_b = self.node_pos[b]
        # y (grandtruth)
        dist = self.cost_matrix[self.hundle_nodes_idx[a], self.hundle_nodes_idx[b]]

        # dist = np.log(dist) + 1
        return (a.astype(np.int64), b.astype(np.int64)), dist.astype(np.float32)



def draw_map(dst, data, labels):
    
    x = data[:, 1]
    y = - data[:, 0]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(x, y, s=50)
    for i in range(len(labels)):
        l = labels[i]
        ax1.annotate(l,xy=(x[i],y[i]),size=10,color="red")
    # plt.show()
    plt.axis('equal')
    plt.savefig(dst)
    plt.clf()
    plt.close()

def draw_image(dst, image, begs, ends):
    tps = cv2.createThinPlateSplineShapeTransformer()

    begs = begs[:, ::-1]
    ends = ends[:, ::-1]
    begs = begs.reshape(1,-1,2)

    mn = np.min(ends)
    mx = np.max(ends)
    # print('mx: ', mx, ', mn: ', mn)

    margin = 25
    dim = 256 - (2 * margin)
    ends = dim * (ends - mn) / (mx - mn) + margin
    ends = ends.reshape(1,-1,2)

    print(begs.shape, ends.shape)

    mn = np.min(ends)
    mx = np.max(ends)
    print('mx: ', mx, ', mn: ', mn)

    # print('begs[:10]: ', begs[0, :10])
    # print('ends[:10]: ', ends[0, :10])


    matches = list()
    for i, _ in  enumerate(begs[0]):
        matches.append(cv2.DMatch(i,i,0))

    
    # tps.estimateTransformation(ends, begs, matches)
    # ret, ends_ = tps.applyTransformation(begs)
    # print(ends[0, :10])
    # print(ends_[0, :10])

    # tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(ends, begs, matches)
    # tps.estimateTransformation(begs, ends, matches)


    deformed = tps.warpImage(image)

    for i, p in enumerate(ends.reshape(-1, 2)):
            p = (int(p[0]), int(p[1]))
            print(p)
            cv2.circle(deformed, p, 1, (255, 0, 0), thickness=-1)
            cv2.putText(deformed, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


    # for i, p in enumerate(ends.reshape(-1, 2)):
    #     p = (int(p[0]), int(p[1]))
    #     cv2.circle(deformed, p, 1, (255, 0, 0), thickness=-1)
            
            

    cv2.imwrite(dst, deformed)
    # return deformed

def get_path(start, goal, pred):
    return get_path_row(start, goal, pred[start])

def get_path_row(start, goal, pred_row):
    path = []
    i = goal
    while i != start and i >= 0:
        path.append(i)
        i = pred_row[i]
    if i < 0:
        return []
    path.append(i)
    return path[::-1]

if __name__ == "__main__":
    n = 100
    embed_dim = 2
    batch_size = 32
    batch_p_ep = 1000
    image_path = './datasets/00002.png'

    train_set = Image_Dataset(image_path, n, batch_size, batch_p_ep)


    # chk = train_set.image.copy()

    # for p in train_set.nodes[:100]:
    #     print(p)
    #     cv2.circle(chk, (p[1], p[0]), 1, (255, 0, 0), thickness=-1)

    # cv2.imshow('chk', chk)
    # cv2.waitKey(0)

    # t = train_set.hundled_nodes[:10]
    # print(t)
    # t = t[:, ::-1]
    # print(t)

    for i in range(10):
        (a, b), dist = train_set.__getitem__(0)
        print('a: ', a, 'b: ', b, 'dist: ', dist)
        a_pos = train_set.hundled_nodes[a]
        b_pos = train_set.hundled_nodes[b]

        a_idx = train_set.hundle_nodes_idx[a]
        b_idx = train_set.hundle_nodes_idx[b]

        path = get_path(a_idx, b_idx, train_set.predecessors)
        print(len(path))
        chk = train_set.image.copy()

        for p in path:
            t = (train_set.nodes[p][1], train_set.nodes[p][0])
            cv2.circle(chk, t, 1, (255, 0, 0), thickness=-1)


        t = a_pos[::-1]
        cv2.circle(chk, t, 2, (0, 0, 255), thickness=-1)
        
        t = b_pos[::-1]
        cv2.circle(chk, t, 2, (0, 0, 255), thickness=-1)

        cv2.imshow('chk', chk)
        cv2.waitKey(0)
                       


    cv2.destroyAllWindows()

    # # ends = train_set.hundled_nodes + 100.0 * np.random.random(2*n).reshape([-1, 2])
    # ends = train_set.hundled_nodes + 20.0

    # draw_image('./outputs/deformed.png', train_set.image, train_set.hundled_nodes, ends)

    



        