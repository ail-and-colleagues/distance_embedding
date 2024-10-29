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

#from t_sne import TSNE_Embedder, calc_dist_mtx
#from t_sne import calc_dist_mtx
from sklearn.metrics import pairwise_distances


from test_dataset import Random_Dataset, draw_map
from model import Dist2Pos

from PIL import Image
import glob
import math

import convert_dataset as con_d 

### calc_dis_mtx() was transported from t_sne.py
def calc_dist_mtx(x):
    return pairwise_distances(x, x, metric='euclidean', n_jobs=-1)
###

def train(st_num):
    #機械学習の処理先指定
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    #出力ディレクトリの指定
    output_dir = datetime.datetime.now().strftime("%Y%m%d_%H_%M_")
    output_dir = os.path.join('./outputs', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # set parameters.
    if st_num == 0:
        node_num = con_d.exsit_2_number('pos')[1]
    else:
        node_num = st_num   #
    pos_dim = 2 #ノードの次元
    embed_dim = 2   #
    batch_size = 64 #
    batch_p_ep = 5120   #
    mx_ep = 50 #

    train_set = Random_Dataset(node_num, pos_dim, batch_size, batch_p_ep)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    
    dist_mtx = calc_dist_mtx(train_set.node_pos)
    print("dist_mtx.shape: ", dist_mtx.shape)
    print(dist_mtx[:5, :5])

    # define model.
    d2p = Dist2Pos(node_num, embed_dim, con_d.exsit_2_number('pos')[0]).to(device)

    criterion = torch.nn.L1Loss()   #MAE,誤差関数を指定
    optimizer = torch.optim.Adam(d2p.parameters(), lr=0.0001)

    # output grandtrutha as gt.png.
    dst = os.path.join(output_dir, 'gt.png')
    draw_map(dst, train_set.node_pos, range(train_set.node_num))

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

                loss = criterion(dist, pred)    #誤差の評価
                loss.backward()             #勾配算出
                optimizer.step()    #反映

                #//--------コンソール表示の設定もろもろ------//

                losses.append(loss.item())

                tqdm_train_loader.set_description("[Epoch %d]" % (ep))
                loss_mean = np.mean(losses)
                tqdm_train_loader.set_postfix(OrderedDict(loss=loss_mean, tloss=loss.item()))
                #//---------ここまで------------------------//

            dst = os.path.join(output_dir, 'pred_{:03d}_{:.5f}.png'.format(ep, loss_mean))
            embed = d2p.fetch().T
            dist_mtx = calc_dist_mtx(embed)
            draw_map(dst, embed, range(train_set.node_num))

            print("_dist_mtx.shape: ", dist_mtx.shape)
            print(dist_mtx[:7, :7])

    rename = output_dir[:-1]
    os.rename(output_dir, rename)

def make_gif_animation(file_path,pic_n,ani_time):
    ani_fps=ani_time/pic_n*1000
    pictures=[]
    path_glob=file_path+'*[0-9].[0-9][0-9][0-9][0-9][0-9].png'
    path_list=glob.glob(path_glob)
    for i in range(pic_n):
        img = Image.open(path_list[i])
        pictures.append(img)
    pictures[0].save(file_path+'\\stpred_all.gif',save_all=True,append_images=pictures[1:],optimize=True,duration=ani_fps,loop=0)

if __name__ == "__main__":
    
    train(0)
    
    defdir='C:\\Users\\yoshi-TKG\\Documents\\VSCode\\outputs\\'
    file='20240920_16_51_\\'
    output_dir=defdir+file
    #make_gif_animation(output_dir,21,1.5)