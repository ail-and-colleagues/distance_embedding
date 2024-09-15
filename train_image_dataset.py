import os
import datetime
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import argparse

import torch
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader

from t_sne import TSNE_Embedder, calc_dist_mtx

from image_dataset import Image_Dataset, draw_map, draw_image

from model import Dist2Pos

parser = argparse.ArgumentParser(description='train embedding')
parser.add_argument('image_path', help='input image path', type=str)

def train(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = datetime.datetime.now().strftime("%Y%m%d_%H_%M_")
    output_dir = os.path.join('./outputs', output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # set parameters.

    embed_dim = 2
    batch_size = 64
    batch_p_ep = 500
    mx_ep = 300
    hundle_nodes_num = 100
    lr = 0.001

    
    train_set = Image_Dataset(image_path, hundle_nodes_num, batch_size, batch_p_ep)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    
    # define model.
    d2p = Dist2Pos(hundle_nodes_num, embed_dim, w=train_set.hundled_nodes / 10.0).to(device)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(d2p.parameters(), lr=lr)

    # output grandtrutha as gt.png.
    dst = os.path.join(output_dir, 'org.png')
    draw_map(dst, train_set.hundled_nodes, range(hundle_nodes_num))


    # t-sne output for comparison.
    # dst = os.path.join(output_dir, 'tsne_pred.png')
    # tsne = TSNE_Embedder(perplexity=5, n_iter=25000, s_random=0)
    # embed = tsne.fit(train_set.node_pos)
    # draw_map(dst, embed, range(train_set.node_num))

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
                a = F.one_hot(a, train_set.hundle_nodes_num)
                b = F.one_hot(b, train_set.hundle_nodes_num)

                # set tensors' gradient to 0 for calcrating and adapting gradients for a current batch at a backward process.
                optimizer.zero_grad()

                # predict positions corresponding to given index a, b.
                emb_a = d2p(a)
                emb_b = d2p(b)
                pred = torch.linalg.norm(emb_b - emb_a, dim=1)

                dist = torch.log(dist) + 1
                pred = torch.log(pred) + 1

                loss = criterion(dist, pred)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                tqdm_train_loader.set_description("[Epoch %d]" % (ep))
                loss_mean = np.mean(losses)
                tqdm_train_loader.set_postfix(OrderedDict(loss=loss_mean, tloss=loss.item()))

            dst = os.path.join(output_dir, 'pred_{:03d}_{:.5f}.png'.format(ep, loss_mean))
            embed, scale = d2p.fetch()
            embed = embed.T
            dist_mtx = calc_dist_mtx(embed)
            draw_map(dst, embed, range(train_set.hundle_nodes_num))

            dst = os.path.join(output_dir, 'deformed_{:03d}_{:.5f}.png'.format(ep, loss_mean))
            draw_image(dst, train_set.image.copy(), train_set.hundled_nodes.copy(), embed)

            print('_dist_mtx.shape: ', dist_mtx.shape)
            print(dist_mtx[:5, :5])
            print('scale: ', scale)

    rename = output_dir[:-1]
    os.rename(output_dir, rename)

if __name__ == "__main__":

    args = parser.parse_args()
    image_path = args.image_path
    train(image_path)