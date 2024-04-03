import os
import datetime
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import matplotlib as mpl

import torch
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader

from t_sne import TSNE_Embedder, calc_dist_mtx

from test_dataset import Random_Dataset, draw_map
from model import Dist2Pos

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = datetime.datetime.now().strftime("%Y%m%d_%H_%M_")
    output_dir = os.path.join('./outputs', output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # set parameters.
    node_num = 10
    pos_dim = 2
    embed_dim = 2
    batch_size = 32
    batch_p_ep = 5000
    mx_ep = 300

    train_set = Random_Dataset(node_num, pos_dim, batch_size, batch_p_ep)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    
    dist_mtx = calc_dist_mtx(train_set.node_pos)
    print("dist_mtx.shape: ", dist_mtx.shape)
    print(dist_mtx[:5, :5])

    # define model.
    d2p = Dist2Pos(node_num, embed_dim, w=None).to(device)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(d2p.parameters(), lr=0.00001)

    # output grandtrutha as gt.png.
    dst = os.path.join(output_dir, 'gt.png')
    draw_map(dst, train_set.node_pos, range(train_set.node_num))

    # t-sne output for comparison.
    dst = os.path.join(output_dir, 'tsne_pred.png')
    tsne = TSNE_Embedder(perplexity=5, n_iter=25000, s_random=0)
    embed = tsne.fit(train_set.node_pos)
    draw_map(dst, embed, range(train_set.node_num))

    # trainig loop.
    for ep in range(mx_ep):
        losses = list()
        with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train_loader:
            for i, ((a, b), dist) in tqdm_train_loader:
                a = a.to(device)
                b = b.to(device)
                dist = dist.to(device)
                # index to one-hot representation. e.g.:
                # id: 1 -> [0, 1, 0, 0, .., 0]
                # id: 3 -> [0, 0, 0, 1, .., 0]
                a = F.one_hot(a, node_num)
                b = F.one_hot(b, node_num)

                # set tensors' gradient to 0 for calcrating and adapting gradients for a current batch at a backward process.
                optimizer.zero_grad()

                # predict positions corresponding to given index a, b.
                emb_a = d2p(a)
                emb_b = d2p(b)
                pred = torch.linalg.norm(emb_b - emb_a, dim=1)

                loss = criterion(dist, pred)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                tqdm_train_loader.set_description("[Epoch %d]" % (ep))
                loss_mean = np.mean(losses)
                tqdm_train_loader.set_postfix(OrderedDict(loss=loss_mean, tloss=loss.item()))

            dst = os.path.join(output_dir, 'pred_{:03d}_{:.5f}.png'.format(ep, loss_mean))
            embed = d2p.fetch().T
            dist_mtx = calc_dist_mtx(embed)
            draw_map(dst, embed, range(train_set.node_num))

            print("_dist_mtx.shape: ", dist_mtx.shape)
            print(dist_mtx[:5, :5])

    rename = output_dir[:-1]
    os.rename(output_dir, rename)

if __name__ == "__main__":
    train()