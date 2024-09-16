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

        # self.create_cost_matrix()
        # npz = np.load(self.cost_matrix_path)


        print('npz.files: ', npz.files)
        self.nodes = npz['nodes']
        self.cost_matrix = npz['cost_matrix']
        self.predecessors = npz['predecessors']

        self.node_num = self.nodes.shape[0]
        print('nodes.shape: ', self.nodes.shape)
        print('cost_matrix.shape: ', self.cost_matrix.shape)
        print('predecessors.shape: ', self.predecessors.shape)


        corner_idxs, self.image = fetch_corner_nodes(self.image, self.nodes)
        
        # self.hundle_nodes_idx = np.random.choice(self.node_num, self.hundle_nodes_num, replace=False)
        self.hundle_nodes_idx = nodes_choice_fps(corner_idxs, self.nodes, self.hundle_nodes_num, self.cost_matrix)
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
        eps = 0.001
        graph = list()
        thres = 2
        for i, p in enumerate(nodes):
            norm = np.linalg.norm(nodes - p, ord=2, axis=1)
            dsts_1 = np.where(norm<1.0+eps, 1, thres+1)
            dsts_r2 = np.where(norm<np.sqrt(2)+eps, np.sqrt(2), thres+1)
            
            dsts = np.minimum(dsts_1, dsts_r2)
            dsts = np.where(dsts < thres, dsts, 0)
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

def nodes_choice_fps(corner_idxs, nodes, num, cost_matrix):
    
    candidates_num = nodes.shape[0]
    sampled = np.zeros((num,))
    distances = np.ones((candidates_num,)) * float('inf')

    sampled[:corner_idxs.shape[0]] = corner_idxs

    for i in corner_idxs:
        t = nodes[i, :]
        d = np.sum((nodes - t) ** 2, -1)
        distances = np.minimum(distances, d)

    # farthest = np.random.randint(0, candidates_num)
    for i in range(corner_idxs.shape[0], num):
        farthest = np.argmax(distances)
        sampled[i] = farthest
        t = nodes[farthest, :]
        d = np.sum((nodes - t) ** 2, -1)
        mask = d < distances
        distances[mask] = d[mask]
        farthest = np.argmax(distances, -1)
        farthest_dist = np.max(distances, -1)

        # d = cost_matrix[farthest]
        # mask = d < distances
        # distances[mask] = d[mask]
        # farthest = np.argmax(distances, -1)

    # farthest_dist = np.sqrt(farthest_dist)
    # print('farthest_dist: ', farthest_dist)

    return sampled.astype(np.int64)

def fetch_corner_nodes(image, nodes):

    bin = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(bin, 127, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 1:
        raise NotImplementedError

    contours = contours[0].reshape([-1, 2])
    contours = contours[:, ::-1]
    

    node_idxs = list()
    for c in contours:
        d = np.linalg.norm(c - nodes, axis=1)
        t = np.argmin(d)
        node_idxs.append(t)

    node_idxs = np.array(node_idxs)
    
    print('node_idxs: ', node_idxs.shape)

    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    image = cv2.bitwise_and(image, mask)

    mask = erode_image(mask, [5, 5])

    image = cv2.bitwise_and(image, mask)

    return node_idxs, image

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

def draw_image(dst_dir, dst_name, image, begs, ends):
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

    farthest_dists = list()
    for p in ends[0]:
        d = ends[0] - p.reshape([1, -1])
        d = np.linalg.norm(d, axis=1)
        d = np.sort(d)
        farthest_dists.append(int(d[1]) + 1)


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



    deformed = tps.warpImage(image, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE, borderValue=(127, 1127, 127))

    for i, p in enumerate(ends.reshape(-1, 2)):
        p = (int(p[0]), int(p[1]))
        cv2.circle(deformed, p, 1, (255, 0, 0), thickness=-1)

    mask = np.zeros_like(deformed)

    for i, p in enumerate(ends.reshape(-1, 2)):
        p = (int(p[0]), int(p[1]))
        cv2.circle(mask, p, farthest_dists[i], (255, 255, 255), thickness=-1)
    cv2.imwrite('./outputs/mask.png', mask)

    # deformed = cv2.bitwise_and(deformed, mask)
    dst = os.path.join(dst_dir, dst_name)                    
    cv2.imwrite(dst, deformed.copy())

    for i, p in enumerate(ends.reshape(-1, 2)):
        p = (int(p[0]), int(p[1]))
        cv2.putText(deformed, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA)
    dst = os.path.join(dst_dir, '_' + dst_name)
    cv2.imwrite(dst, deformed)

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

    image = cv2.imread(image_path)
    int_area = image[:, :, 2] != 127
    wall =  image[:, :, 2] != 0
    mask = int_area & wall

    mask = np.where(mask)
    nodes = np.concatenate([mask[0].reshape([-1, 1]), mask[1].reshape([-1, 1])], axis=1)

    # chk
    t = np.zeros_like(image)
    t[mask] = 255
    cv2.imshow('t', t)
    cv2.waitKey(0)

    bin = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(bin, 127, 255, cv2.THRESH_BINARY)

    # contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # img_contour_only = cv2.drawContours(image, contours, -1, (0,255,0), 1)

    _contours = contours[0].reshape([-1, 2])
    print('contours: ', _contours.shape)
    for i, p in enumerate(_contours):
        # p = (int(p[0]), int(p[1]))
        cv2.circle(image, p, 1, (255, 0, 0), thickness=-1)
    
    # print(contours[0].shape)
    cv2.imshow('t', image)
    cv2.waitKey(0)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    image = cv2.bitwise_and(image, mask)
    cv2.imshow('t', image)
    cv2.waitKey(0)  

    # train_set = Image_Dataset(image_path, n, batch_size, batch_p_ep)


    # # chk = train_set.image.copy()

    # # for p in train_set.nodes[:100]:
    # #     print(p)
    # #     cv2.circle(chk, (p[1], p[0]), 1, (255, 0, 0), thickness=-1)

    # # cv2.imshow('chk', chk)
    # # cv2.waitKey(0)

    # # t = train_set.hundled_nodes[:10]
    # # print(t)
    # # t = t[:, ::-1]
    # # print(t)

    # for i in range(10):
    #     (a, b), dist = train_set.__getitem__(0)
    #     print('a: ', a, 'b: ', b, 'dist: ', dist)
    #     a_pos = train_set.hundled_nodes[a]
    #     b_pos = train_set.hundled_nodes[b]

    #     a_idx = train_set.hundle_nodes_idx[a]
    #     b_idx = train_set.hundle_nodes_idx[b]

    #     path = get_path(a_idx, b_idx, train_set.predecessors)
    #     print(len(path))
    #     chk = train_set.image.copy()

    #     for p in path:
    #         t = (train_set.nodes[p][1], train_set.nodes[p][0])
    #         cv2.circle(chk, t, 1, (255, 0, 0), thickness=-1)


    #     t = a_pos[::-1]
    #     cv2.circle(chk, t, 2, (0, 0, 255), thickness=-1)
        
    #     t = b_pos[::-1]
    #     cv2.circle(chk, t, 2, (0, 0, 255), thickness=-1)

    #     cv2.imshow('chk', chk)
    #     cv2.waitKey(0)
                       


    # cv2.destroyAllWindows()

    # # ends = train_set.hundled_nodes + 100.0 * np.random.random(2*n).reshape([-1, 2])
    # ends = train_set.hundled_nodes + 20.0

    # draw_image('./outputs/deformed.png', train_set.image, train_set.hundled_nodes, ends)

    



        