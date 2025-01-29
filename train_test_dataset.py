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
from torch.optim import lr_scheduler

#from t_sne import TSNE_Embedder, calc_dist_mtx
#from t_sne import calc_dist_mtx
from sklearn.metrics import pairwise_distances


from test_dataset import Random_Dataset, draw_map
from model import Dist2Pos

from PIL import Image
import glob
import math

import itertools
import convert_dataset as con_d 
import time
t_start=time.time()

### calc_dis_mtx() was transported from t_sne.py
def calc_dist_mtx(x):
    return pairwise_distances(x, x, metric='euclidean', n_jobs=-1)
###

def train(seed=0,
          fc_seed=0,
          preview=False,
          memo="",
          show=True,
          ep_m=80,
          pos_dim=2,
          node_exs=False,
          pos_exs=False,
          dist_exs=False):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    new_dir = con_d.dir_ct(-1).append_file()
    os.makedirs(new_dir, exist_ok=True)
    output_dir = con_d.dir_ct().append_file()
    
    node_tf_data=con_d.datas(
        seed=seed,
        fc_seed=fc_seed,
        node_tf_exs=node_exs,
        pos_exs=pos_exs,
        dist_exs=dist_exs,
        dim=pos_dim)
    
    node_pos = node_tf_data.get_pos()
    node_num = node_tf_data.get_node_n()
    node_num_sum=node_num[0]
    dist_mtx = node_tf_data.get_dist()
    dist_mtx = con_d.mat_connect(node_tf_data, dist_mtx)
    node_color = con_d.pos_2_color(node_tf_data)
    
    for _,map_view in enumerate(seed):
        map_xy=node_pos[map_view]
        map_n=node_tf_data.get_node_n()[2][map_view]
        con_d.draw_map(map_xy,map_n,show=show,save=True,seed_num=map_view)
    
    embed_dim = 2
    batch_size = 64
    batch_p_ep = 512
    mx_ep = ep_m 

    for i in range(len(node_pos)):
        node_pos_c=np.vstack([node_pos_c,node_pos[i]]) if i!=0 else node_pos[i]

    train_set = Random_Dataset(node_num_sum, pos_dim, batch_size, batch_p_ep, dist_mtx, node_pos_c)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    
    #dist_mtx = calc_dist_mtx(train_set.node_pos)
    #print("dist_mtx.shape: ", dist_mtx.shape)
    #print(dist_mtx[:5, :5])

    # define model.
    #d2p = Dist2Pos(node_num, embed_dim).to(device)
    d2p = Dist2Pos(node_num_sum, embed_dim, train_set.node_pos).to(device)

    criterion = torch.nn.L1Loss()   #MAE,誤差関数を指定
    optimizer = torch.optim.Adam(d2p.parameters(), lr=0.005)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=math.floor(mx_ep/5), eta_min=0.0001)

    # output grandtrutha as gt.png.
    dst = os.path.join(output_dir, 'gt.png')
    draw_map(dst, train_set.node_pos, range(train_set.node_num_sum),node_color)

    # trainig loop.
    for ep in range(mx_ep):
        losses = list()
        with tqdm(enumerate(train_loader), total=len(train_loader),leave=False) as tqdm_train_loader:
            for i, ((a, b), dist) in tqdm_train_loader:
                a = a.to(device)
                b = b.to(device)
                dist = dist.to(device)
                # index to one-hot representation. e.g.:
                # id: 1 -> [0, 1, 0, 0, .., 0]
                # id: 3 -> [0, 0, 0, 1, .., 0]
                a = F.one_hot(a, node_num_sum)
                b = F.one_hot(b, node_num_sum)

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
            draw_map(dst, embed, range(train_set.node_num_sum),node_color)
            csv_txt="{:0=3}th_pos.txt".format(ep)
            #np.savetxt(os.path.join(output_dir,csv_txt),embed,fmt='%.2f',delimiter='\t')

            if (preview):
                print("_dist_mtx.shape: ", dist_mtx.shape)
                print(dist_mtx[:7, :7])
        #scheduler.step()

    memo = memo if memo != "" else "completed"
    dir_temp=os.path.join(os.path.dirname(__file__),"outputs")
    rename = "test{:0=3}__{}".format(con_d.dir_ct(0).dir_n,memo)
    os.rename(output_dir, os.path.join(dir_temp,rename))

def make_gif_animation(file_n=0,pic_n=1,ani_time=2.5,ani_fps=None):
    file_n = con_d.dir_ct(file_n).dir_n
    print("generating {:0=3} th file...".format(file_n))
    path = con_d.dir_ct(file_n).dir_nth_match()
    img_match='*[0-9].[0-9][0-9][0-9][0-9][0-9].png'
    path_list=glob.glob(os.path.join(path,img_match))
    pic_n=len(path_list)
    ani_fps=ani_fps if type(ani_fps)==int else ani_time/pic_n*1000
    pictures=[]
    for i in range(pic_n):
        img = Image.open(path_list[i])
        pictures.append(img)
    pictures[0].save(path+'\\stpred_all.gif',save_all=True,append_images=pictures[1:],optimize=True,duration=ani_fps,loop=0)
    print("animation generating has accomplished")

def profile_test():
    import cProfile
    import pstats

    cProfile.run('train(output_dir,seed,memo=memo,ep_m=5)','profiling_resurt')
    p=pstats.Stats('profiling_resurt')
    p.sort_stats(pstats.SortKey.FILENAME).print_stats()

if __name__ == "__main__":
    
    tr=1
    
    seed = (1,1)
    fc_seed = range(2,14)
    
    len_seed=len(seed)
    node_tf_exs=[False]*len_seed
    pos_exs=[False]*len_seed
    dist_exs=[False]*len_seed
    ep_m = 1
    #memo = "{}f_2p_cs{:0=2}".format(len(seed),fc_seed)
    
    #con_d.map_preview(seed)
    
    ani_tr=False #generating gif animation after training.
    ani_time=3  #animation time length.
    ani_file_n=0    #how many pictures use when animation generating. if '0' , all pictures in processing file will be used.

    def_dir=os.path.join(os.path.dirname(__file__),"outputs")
    #output_dir=con_d.rt_dir(def_dir)
    for tr_loop in fc_seed:
        output_dir=con_d.dir_ct(-1).append_file()
        memo = "dist_mtx_{}f_2p_cs{:0=2}".format(len(seed),tr_loop)
        if (tr):
            #n=con_d.exsit_2_number('pos',seed)[1]
            train(seed=seed, fc_seed=tr_loop, memo=memo, show=False,
                ep_m=ep_m, pos_dim=2, node_exs=node_tf_exs, pos_exs=pos_exs,
                dist_exs=dist_exs)
        
        if (0): #profiling
            profile_test()

        if (ani_tr):
            ani_file_n = list([ani_file_n]) if type(ani_file_n)!=list else ani_file_n
            for _,i in enumerate(ani_file_n):
                #ani_dir=con_d.dir_ct(i).dir_nth_match()
                make_gif_animation(file_n=i,ani_time=ani_time)

t_end=time.time()
print(t_end-t_start)