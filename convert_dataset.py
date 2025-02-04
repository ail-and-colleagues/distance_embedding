#import torch
#from torch.utils.data import Dataset

import numpy as np
import matplotlib  # <--ここを追加
matplotlib.use('Agg')  # <--ここを追加
from matplotlib import pyplot as plt
import os
import itertools

import datetime

import glob
import re

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Manager
import copy

f_inf = np.inf

class network_test():
    import networkx 
    nx = networkx
    def __init__(self,node_name,edge_con,edge_w,nx=nx) :
        #self.a=a        
        self.G=0
        self.node_name=node_name
        self.edge_con=edge_con
        self.edge_w=edge_w
        self.nx=nx

    def test(self):
        self.nx.draw(self.G)

    def set(self):
        Gh=self.nx.Graph()
        for i_node in range(len(self.node_name)):
            Gh.add_node(self.node_name[i_node])
        for i_edge in range(len(self.edge_con)):
            Gh.add_edge(self.edge_con[i_edge][0],self.edge_con[i_edge][1],weight=1/self.edge_con[i_edge][2])
        self.nx.draw(Gh, with_labels=True)
        plt.show()

class dir_ct: #direcory_control
    def __init__(self,n=0):
        #dir_n:0なら既存フォルダ数を自動集計、-1なら既存数+1(新しくディレクトリを作成する用)
        def dir_count(n_mode=0):
            file_list=glob.glob(os.path.join(self.def_dir,'**/'))
            dir_n=0
            for i in range(len(file_list)):
                file_list[i]=file_list[i].removeprefix(self.def_dir+'\\')
                if file_list[i].startswith("test"):
                    dir_n = dir_n + 1
            return(dir_n-n_mode)
    
        self.def_dir = os.path.join(os.path.dirname(__file__),"outputs")
        self.dir_n = n if n > 0 else dir_count(n)
        
        self.dir_str_def = "test{:0=3}__".format(self.dir_n)
        self.dir_pros = self.dir_str_def + "processing"
        self.dir_wild = self.dir_str_def + "*"

    def append_file(self):
        return(os.path.join(self.def_dir,self.dir_pros))    

    def dir_nth_match(self):
        file_title = self.dir_str_def
        file_title += '*'
        file_nth_path=glob.glob(os.path.join(self.def_dir,file_title))
        return(file_nth_path[0])

class datas:
    def __init__(self,
                 seed=0,
                 fc_seed=0,
                 node_tf_exs=False, #node_tfの外部ファイル有無
                 pos_exs=False,  #posの外部ファイル有無
                 dist_exs=False,    #distの外部ファイル有無s
                 dim=2  #点の次元
                 ):
        
        def return_param_dict(seed_any):
            #引数をdict型に変換
            seed_dic={}
            if type(seed_any)==dict:
                seed_dic.update(seed_any)
            elif type(seed_any)==list or type(seed_any)==tuple:
                for k,v in enumerate(seed_any):
                    seed_dic[k]=v
            else:
                seed_dic[0]=seed_any
            return seed_dic

        def get_dataset(node_tf_dic,seed_dic):
            nodes=[]
            for seed_n,seed_value in seed_dic.items():
                node_tf_value=node_tf_dic[seed_n]

                #どちらの関数も引数はseedの値
                if(node_tf_value): #外部ファイルから読み取り
                    nodes.append(import_data(seed_dic[seed_n],mode='node_tf'))
                else:   #get_node_tf()内に記述されたノード情報を持ってくる
                    nodes.append(get_node_tf(seed_dic[seed_n]))
            return nodes

        self.dim = dim
        self.fc_seed = fc_seed
        self.seed   = return_param_dict(seed)
        self.node_tf_exs= return_param_dict(node_tf_exs)
        self.pos_tf_exs = return_param_dict(pos_exs)
        self.dist_exs = return_param_dict(dist_exs)
        
        self.seed_length = len(self.seed)

        self.node_tf = get_dataset(self.node_tf_exs,self.seed)
        self.sc_path=os.path.dirname(__file__)

    def get_node_n(self):   #seed毎のノード数取得
        all_sum=0
        ch_sum=[0]
        ch=[]
        for i in range(self.seed_length):
            current_node_t=self.node_tf[i]
            count_temp=len(sum(np.where(current_node_t >= 1)))
            ch.append(count_temp)
            all_sum+=count_temp
            ch_sum.append(all_sum)
        
        return all_sum,ch_sum,ch

    def get_pos(self):  #点の座標呼び出し
        pos_data=[]
        for key,pos_tf_value in self.pos_tf_exs.items():
            nth_seed = key
            if pos_tf_value:
                pos_data.append(import_data(nth_seed,mode='node_pos'))
            else:
                pos_data.append(node_tf2pos(nth_seed,self.dim,self.node_tf[key]))
        return pos_data
    
    def get_dist(self):  #時間行列の呼び出し
        dist_data=[]
        for key,dist_tf_value in self.dist_exs.items():
            nth_seed = self.seed[key]
            if dist_tf_value:
                dist_data.append(import_data(nth_seed,mode='dist_mat'))
            else:
                dist_data.append(make_dist(nth_seed,self.node_tf[key]))
        return dist_data

def import_data(seed_value=0,mode='node_tf'):
        #mode=('node_tf','node_pos','dist_mat')
    try:
        path=os.path.join(os.path.dirname(__file__),'imputs',mode,'{:0=3}.txt'.format(seed_value))
    except:
        print("\n\n /import/{mode} 内に{seed_value}.txtが存在しません。\n\n")
        path=os.path.join(os.path.dirname(__file__),'imputs',mode,'default.txt'.format(seed_value))
    
    with open(path) as f:
        line=f.read().splitlines()
    line=line[1] if line[1] else line[0]
    #pt=re.compile(r'(\D[^\.])')
    #delimiter=re.search(pt,line).group(0)
    delimiter = "\t"
    i_data=np.loadtxt(path,delimiter=delimiter)
    print(i_data)
    return i_data

def node_tf2pos(seed_nth,td,node_tf):
    #td: 3rd dimmension   
    if td==3:
        z=np.full(len(np.where(node_tf>0)[0]),seed_nth)
    tf_where=np.where(node_tf>=1)
    y_max=np.amax(tf_where[1])
    x=tf_where[1]
    y=np.where(tf_where[0] >= 0 ,y_max-tf_where[0],0)
    if td==2:
        pos_data=np.array(list(zip(x,y)))
    else:
        pos_data=np.array(list(zip(x,y,z)))
    return pos_data

def make_dist(seed_nth,node_tf,unit_h=1,unit_v=1):
    pos=node_tf2pos(seed_nth,2,node_tf)
    unit_diag=np.sqrt(unit_h**2 + unit_v**2)
    node_n = len(pos)
    dist_mat=np.where(np.eye(node_n)==0,f_inf,0)

    for i in range(node_n-1):
        x_i,y_i=pos[i][0],pos[i][1]
        for j in range(i+1,node_n):
            x_j,y_j=pos[j][0],pos[j][1]
            pos_dif_x=abs(x_j-x_i)
            pos_dif_y=abs(y_j-y_i)
            if pos_dif_x*pos_dif_y == 1:
                dist_mat[i,j] = unit_diag
                dist_mat[j,i]=dist_mat[i,j]
            elif pos_dif_x+pos_dif_y == 1:
                dist_mat[i,j] = unit_h
                dist_mat[j,i]=dist_mat[i,j]
    return dist_mat

def wf_parallel(k,n,dd):
    #dd=dd.copy()
    #dd_copy = copy.deepcopy(dd)
    for i in range(n):
        for j in range(n):
            dd[i][j] = min(dd[i][j],dd[i][k]+dd[k][j])
    print("k:{:0=4}/{}".format(k,n))
    return dd

def mat_connect(cl, dist_data):
    pos_n, fc_seed = cl.get_node_n(), cl.fc_seed
    p_sum , p_ele = pos_n[1] , pos_n[2]
    cost_def=1.2
    mat_len = len(dist_data)    

    def wf(dd):
        n = len(dd)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dd[i,j] = min(dd[i,j],dd[i,k]+dd[k,j])
            print("{:0=3}/{}".format(k,n))
        dist_out(n,dd)
        return dd


    for y in range(mat_len):
        for x in range(mat_len):
            h_temp = np.full((p_ele[y],p_ele[x]),f_inf) if y!=x else dist_data[y]
            h = np.hstack([h,h_temp]) if x!=0 else h_temp
        v=np.vstack([v,h]) if y!=0 else h

    if mat_len > 1: #接続点の処理
        floor_connector=get_floor_connection(fc_seed)
        fp , co = floor_connector[0] , floor_connector[1]   #floor_connect_point, cost
        fp_len , floor_n = len(fp) , len(fp[0])   #接続点数, フロア数
        for i in range(fp_len):
            try:
                co_n=len(co[i])
                if floor_n - co_n >1:
                    cost_def=co[i][co_n-1]
                    co[i].extend([cost_def]*(floor_n-co_n-1))
            except IndexError:
                if i == 0 :
                    co.append([cost_def]*(floor_n-1))
                else:
                    co.append(co[i-1])
            j1=0
            while j1 < floor_n-1:
                j2 = j1+1
                if fp[i][j1]==-1:
                    j1 += 1
                    continue
                elif fp[i][j2]==-2:
                    j2 += 1
                v[p_sum[j1]+fp[i][j1],p_sum[j2]+fp[i][j2]]=co[i][j1]
                v[p_sum[j2]+fp[i][j2],p_sum[j1]+fp[i][j1]]=co[i][j1]
                j1=j2
    #return wf(v)
    n=len(v)
    #with ThreadPoolExecutor() as executor:
    #    executor.map(lambda k :wf_parallel(k,n,v),range(n))
    #with Manager() as manager:
    #    v_s=manager.list(v.tolist())
    #    with Pool() as pool:
    #        pool.starmap(wf_parallel, [(k, n, v_s) for k in range(n)])
    #    v=np.array(v_s)
    v = wf(v)
    #np.savetxt(os.path.join(dir_ct().append_file(),"dist_matrix.txt"),v,fmt='%.2f',delimiter='\t')
    return v

#------------削除予定---------------#
def _old():
    def convert_exist2pos(unit_h,unit_v,*exist_mat):
        i_rt=0
        if type(exist_mat)==tuple:
            i_rt=len(exist_mat)
            print(exist_mat)
        else:
            i_rt=1
            exist_mat=np.array([exist_mat])
        i=0
        pos_mat=np.arange(2).reshape(1,2)
        for i1 in range(i_rt):
            mat_x = exist_mat[i1][0].shape[1]
            mat_y = exist_mat[i1][0].shape[0]
            for y in range(mat_y):
                for x in range(mat_x):
                    if(exist_mat[i1][0][y,x]==1):
                        posi = np.array([(-mat_x+x)*unit_h,(mat_y-y)*unit_v]).reshape(1,2)
                        pos_mat=np.vstack((pos_mat,posi))
                        i += 1
        #pos_mat=pos_mat.reshape(i,2)
        pos_mat=np.delete(pos_mat,0,axis=0)
        return(pos_mat,i)   #get_pos_numでiは取得可なのでいらないかもしれない

    def get_pos_num(*exist_mat):
        if type(exist_mat)==tuple:
            i_rt=len(exist_mat)
        else:
            i_rt =1
        all_sum=0
        ch_sum=[0]
        ch=[]
        for i1 in range(i_rt):
            tmp=0
            mat_x = exist_mat[i1][0].shape[1]
            mat_y = exist_mat[i1][0].shape[0]
            for y in range(mat_y):
                for x in range(mat_x):
                    if(exist_mat[i1][0][y,x]==1):
                        tmp += 1
            ch.append(tmp)
            all_sum=all_sum+tmp
            ch_sum.append(all_sum)
        return(all_sum,ch_sum,ch)

    def adj_node_tf(n,max):
        if 0<n<max:
            adj_range=np.array([-1,0,1])
        elif n==0:
            adj_range=np.array([0,1])
        else:
            adj_range=np.array([-1,0])
        return adj_range

    def make_dist_mat(unit_h,unit_v,*node_mat):
        if type(node_mat)==tuple:
            i_rt=len(node_mat)
        else:
            i_rt=1
            node_mat=np.array([node_mat])
        unit_diag = np.sqrt(unit_h**2+unit_v**2)    #斜め移動の距離
        mat_p_num=get_pos_num(*node_mat)
        rt_mat=[]
        for n in range(i_rt):
            mat_x = node_mat[n][0].shape[1]   #行列の横方向要素数
            mat_y = node_mat[n][0].shape[0]   #行列の縦方向要素数
            mat_temp=np.identity(mat_y*mat_x)   #(x*y)次元の単位行列を生成
            con_mat = np.where(mat_temp==0,f_inf,0).astype(np.float32)
            for y in range(mat_y):  #n行目の処理
                y_temp_range=adj_node_tf(y,mat_y-1)
                for x in range(mat_x):  #m列目の処理
                    x_temp_range=adj_node_tf(x,mat_x-1)
                    if node_mat[n][0][y,x]!=0: #node=trueのマスの時    
                        for y_temp in y_temp_range:
                            for x_temp in x_temp_range:
                                if(node_mat[n][0][y+y_temp,x+x_temp]==1):  #隣接マスがtrueのとき
                                    if(x_temp*y_temp!=0):
                                        con_mat[(mat_x)*y+x,(mat_x)*(y+y_temp)+x+x_temp]=unit_diag
                                    elif(x_temp==0 and y_temp==0):
                                        con_mat[(mat_x)*y+x,(mat_x)*(y+y_temp)+x+x_temp]=0
                                    elif(x_temp==0):
                                        con_mat[(mat_x)*y+x,(mat_x)*(y+y_temp)+x+x_temp]=unit_v
                                    elif(y_temp==0):
                                        con_mat[(mat_x)*y+x,(mat_x)*(y+y_temp)+x+x_temp]=unit_h
                    else:
                        con_mat[(mat_x)*y+x,0]=-1
            for i in range(mat_x*mat_y-1,-1,-1):
                if con_mat[i,0]==-1:
                    con_mat=np.delete(np.delete(con_mat,i,0),i,1)
            rt_mat.append(con_mat)
        if i_rt==1:
            rt_data=warshal_floyd(mat_p_num[0],*rt_mat)
        else:
            rt_data=dist_mat_connect(mat_p_num,*rt_mat)
        return(rt_data)

    def dist_mat_connect(mat_p_num,*dist_mat):
        A = len(dist_mat)  #引数の数    
        pos_num_c_ele=np.empty(A,dtype=np.uint16)
        
        pos_num_sum = mat_p_num[0]
        pos_num_c_sum=mat_p_num[1]
        pos_num_c_ele=mat_p_num[2]

        print("...combining distance matrixes...")
        #for i in range(A):
        #    pos_num_c_ele[i]=dist_mat[i].shape[0]  #i番目のdist_matのサイズ取得
        mat_zero=[]
        
        for i2 in range(A):
            mat_zero.append([])
            for j2 in range(A):
                if(i2!=j2):
                    mat_zero[i2].append(np.full((pos_num_c_ele[i2],pos_num_c_ele[j2]),f_inf))
                else:
                    mat_zero[i2].append(dist_mat[i2])
            mat_rt_htemp = mat_zero[i2][0]
        
            for k2 in range(1,A):
                mat_rt_htemp=np.hstack([mat_rt_htemp,mat_zero[i2][k2]])
            con_mat=np.vstack([con_mat,mat_rt_htemp]) if i2!=0 else mat_rt_htemp

        #for i3 in range(A):
        
        connection = get_train_dataset('connection')[0]
        weight = get_train_dataset('connection')[1]
        con_n_def = len(connection)

        #connectionの処理---
        
        #-------------------
        
        #if con_n_def!=len(weight):
        #    print()
        #    weight=list(weight[0] * con_n_def)
        #    print('!!!CAUTOIN!!!')
        #    print('conection weight list is shortage.')
        for i4 in range(con_n_def):
            if i4!=0:
                if not(weight[i4]):
                    weight.append(weight[i4-1])

            j4=0

            while j4 < len(weight[i4]):
                k4=j4+1
                if weight[i4][j4]==-1:
                    continue
                elif weight[i4][j4]==-2:
                    k4 += 1
                con_mat[pos_num_c_sum[j4]+connection[i4][j4],pos_num_c_sum[k4]+connection[i4][k4]]=weight[i4][j4]
                con_mat[pos_num_c_sum[k4]+connection[i4][k4],pos_num_c_sum[j4]+connection[i4][j4]]=weight[i4][j4]
                j4=k4

        return(warshal_floyd(pos_num_sum,con_mat))
        
    def warshal_floyd(mat_num_max,con_mat):
        con_mat=np.array(con_mat)
        print("...start  to calculate time distance...")
        for k in range(mat_num_max):
            for i in range(mat_num_max):
                for j in range(mat_num_max):
                    con_mat[i,j] = min(con_mat[i,j],con_mat[i,k]+con_mat[k,j])
        for l in range(mat_num_max-1,0,-1):
            if con_mat[0,l]==f_inf:
                con_mat=np.delete(np.delete(con_mat,l,0),l,1)
        if not(__name__=="__main__"):
            dist_out(mat_num_max,con_mat)    
        return(con_mat,0)
    
    def get_train_dataset(mode,seed=0):
        rt2=0
        if mode=='node_pos':  #各点の位置
            if (seed==0):
                rt_data=np.array([
                    [1,1],
                    [2,2],
                    [1,2],
                    [2,1],
                    [1,3],
                    [1,2],
                    [2,2]
                    ])
                rt2=rt_data.shape[0]
        elif mode=='distance':  #点同士の所要時間
            if (seed==0):
                rt_data =np.array([
                    [ 0, 1, 2, 3 ,0 ,1 ,2],   #1,1
                    [ 1, 0, 1, 2, 1, 2, 3],   #2,2
                    [ 2, 1, 0, 1, 2, 3, 4],   #1,2
                    [ 3, 2, 1, 0, 3, 4, 5],   #2,1
                    [ 0, 1, 2, 3, 0, 1, 2],   #2,3
                    [ 1, 2, 3, 4, 1, 0, 1],   #3,2
                    [ 2, 3, 4, 5, 2, 1, 0]   #3,3
                    ])
        elif mode=='edge': #ノード同士の結び方
            if (seed==0):
                rt_data = np.array([[1,2],
                                    [0,3],
                                    [0,3],
                                    [1,2]])
        elif mode=='node_exist':
            if (seed==0):
                rt_data = np.array([
                    [1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,0,0,1,1,0,0,1,1],
                    [1,1,1,0,0,1,1,0,0,1,1],
                    [1,1,1,0,0,1,1,0,0,1,1]
                    ]).astype(np.uint8)
            elif(seed==1):
                rt_data=np.array([
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1]
                    ]).astype(np.uint8)
            elif(seed==2):
                rt_data=np.array([
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,0,1,0,0,1],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0]
                    ]).astype(np.uint8)
            elif(seed==3):
                rt_data=np.array([
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,1],
                    [1,0,1,0,0,1]
                    ]).astype(np.uint8)
            #9*9glid
        elif mode=='connection':
            if (seed==0):
                rt_data=[[5,5]]
                rt2=[[1.2]]
        return (rt_data,rt2)

    #@lru_cache(maxsize=None)
    def exsit_2_number(mode,seed=0,path=''):
        h=1
        v=1

        if(path=='tf_p'):
            with open(path) as f:
                line=f.read().splitlines()
            line=line[1]
            pt=re.compile(r'(\D)')
            delimiter=re.search(pt,line).group(0)
            rt1=np.loadtxt(path,delimiter=delimiter)
            rt2=0
            exs=list([rt1,rt2])
            print(exs)
            #この後のexsの処理を要調整
        else:
            if type(seed)==int :
                print("int")
                exs=list([get_train_dataset('node_exist',seed)])
            elif type(seed)==list:
                rp=len(seed)
                exs=[]
                for _ , key1 in enumerate(seed):
                    exs.append(list(get_train_dataset('node_exist',key1)))
            else:
                print("other")
                exs=get_train_dataset('node_exist',seed)

        rt_data=None

        if(mode=='pos'):
            #if len(exs)==1:
            #    rt_data=convert_exist2pos(h,v,exs)
            #    #rt_data=get_train_dataset('node_pos')
            #else:
            if path=='pos_p':
                None
            else:
                rt_data=convert_exist2pos(h,v,*exs)
        elif (mode=='dis'):
            #if len(exs)==1:
            #    rt_data=make_dist_mat(h,v,exs)
            #    #rt_data=get_train_dataset('distance')
            #else:
            rt_data=make_dist_mat(h,v,*exs)

        return(rt_data)

def dist_out(mat_num_max,con_mat):
    n = np.arange(mat_num_max)
    out_mat = np.vstack([n,con_mat])
    n = np.insert(n,0,-1).reshape(-1,1)
    out_mat = np.hstack([n,out_mat])
    output_dir=os.path.join(dir_ct(0).append_file(),'dist_matrix.txt')

    np.savetxt(output_dir,out_mat,fmt='%.2f',delimiter='\t')

def get_node_tf(seed:int):
    if (seed==0):
        rt_data = np.array([
            [1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,0,0,1,1,0,0,1,1],
            [1,1,1,0,0,1,1,0,0,1,1],
            [1,1,1,0,0,1,1,0,0,1,1]
            ]).astype(np.uint8)
    elif(seed==1):
        rt_data=np.array([
            [1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1]
            ]).astype(np.uint8)
    elif(seed==2):
        rt_data=np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            ]).astype(np.uint8)
    elif(seed==3):
        rt_data=np.array([
            [1,0,0,1,0],
            [1,0,1,0,0],
            [1,1,1,1,1],
            [1,1,0,0,0]
            ]).astype(np.uint8)
    elif(seed==4):
        rt_data=np.array([
            [1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1]
            ]).astype(np.uint8)
    elif(seed==5):
        rt_data=np.array([
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,0,0,0,0],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
            ]).astype(np.uint8)
    elif(seed==6):
        rt_data=np.array([
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,0,0,1,1],
            [1,1,0,0,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
            ]).astype(np.uint8)
    elif(seed==7):
        rt_data=np.array([
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,0,0,0,0,0],
            [1,0,1,1,1,1],
            [1,0,1,1,0,1],
            [1,0,1,1,0,1],
            [1,0,1,1,0,1],
            [1,0,0,0,0,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
            ]).astype(np.uint8)


    return rt_data

def get_floor_connection(c):
    if (c==0):
        c_p=[[0,0]]
        cost=[[1.2]]
    elif (c==1):
        c_p=[[0,0,0],[5,-2,5],[-1,3,3]]
        cost=[[1.2,1.2],[1.2]]
    elif (c==2):
        c_p=[[0,0]]
        cost=[[1.2]]
    elif (c==3):
        c_p=[[3,3]]
        cost=[[1.2]]
    elif (c==4):
        c_p=[[10,10]]
        cost=[[1.2]]
    elif (c==5):
        c_p=[[0,0],[6,6]]
        cost=[[1.2]]
    elif (c==6):
        c_p=[[0,0],[13,13]]
        cost=[[1.2]]
    elif (c==7):
        c_p=[[0,0],[20,20]]
        cost=[[1.2]]
    elif (c==8):
        c_p=[[7,7],[13,13]]
        cost=[[1.2]]
    elif (c==9):
        c_p=[[0,0],[10,10]]
        cost=[[1.2]]
    elif (c==10):
        #上辺3つ
        c_p=[[0,0],[3,3],[6,6]]
        cost=[[1.2]]
    elif (c==11):
        #対角線
        c_p=[[0,0],[10,10],[20,20]]
        cost=[[1.2]]
    elif (c==12):
        #中央列3つ
        c_p=[[7,7],[10,10],[13,13]]
        cost=[[1.2]]
    elif (c==13):
        #V字状
        c_p=[[0,0],[6,6],[17,17]]
        cost=[[1.2]]
    else :
        c_p,cost=[[0,0]],[[1.2]]
    
    
    return [c_p,cost]



#@lru_cache(maxsize=None)
def set_map_color():
    def_color_set=plt.get_cmap("tab20")
    c_even=[]
    c_odd=[]
    for i in range(def_color_set.N):
        c_even.append(def_color_set(i)) if i%2==0 else c_odd.append(def_color_set(i))
    return(c_even,c_odd)

def pos_2_color(cl):
    #ch:node_num_child
    ch , fc_seed =cl.get_node_n()[2], cl.fc_seed

    color_set=set_map_color()
    color_c=color_set[0]
    color_f=color_set[1]
    c_p=get_floor_connection(fc_seed)[0]
    for i in range(len(ch)):
        c_temp=[color_f[i]]*ch[i]
        for j in range(len(c_p)):
            node=c_p[j][i]
            c_temp[node] = color_c[j] if node>=0 else c_temp[node]
        c_set = np.vstack([c_set,c_temp]) if i!=0 else np.array(c_temp)
    return(c_set)

def draw_map(data,labels,show=True,save=False,seed_num=0):
    x=data[:,0]
    y=data[:,1]
    fig,ax1=plt.subplots(1,1)
    
    for i in range(labels):
        ax1.plot(x[i],y[i],'o',ms=10,mfc='b',mew=0)
        ax1.annotate(i,xy=(x[i],y[i]),size=15,color='r')
        # plt.show()
    plt.axis('equal')
    #fig.canvas.mpl_connect('pick_event',onClick)
    if show:
        plt.show()
    if save:
        dir=dir_ct(0).append_file()
        dir=os.path.join(dir,"def_seed{:0=2}_pos.png".format(seed_num))
        plt.savefig(dir)
    plt.close()

def map_preview(seed):
    seed = seed if type(seed)==list else list([seed])
    for _,map_view in enumerate(seed):
        map_temp=exsit_2_number('pos',map_view)
        map_xy=map_temp[0]
        map_n=map_temp[1]
        draw_map(map_xy,map_n,show=True,seed_num=map_view)


if __name__=="__main__":
    #dir=os.path.join(os.path.dirname(__file__),"imputs","node_exist.txt")
    #exsit_2_number('pos',path=dir)
    #exsit_2_number('dis',seed=[0,1,0])
    output_dir = dir_ct().def_dir
    
    seed=(0,1,2)
    fc_seed=1
    node_tf_exs=[False,False,0]
    pos_exs=[0,0,0]
    dist_exs=[0,0,0]
    node_tf_data=datas(
        seed=seed,
        fc_seed=fc_seed,
        node_tf_exs=node_tf_exs,
        pos_exs=pos_exs,
        dist_exs=dist_exs,
        dim=2)
    node_n=node_tf_data.get_node_n()
    node_dist=node_tf_data.get_dist()
    dist_mat=mat_connect(node_tf_data, node_dist)
    color_mat=pos_2_color(node_tf_data)
    #np.savetxt(os.path.join(output_dir,"mat_l_dist.txt"),dist_mat,fmt='%.2f',delimiter='\t')
    print(color_mat)