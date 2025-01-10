import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib  # <--ここを追加
matplotlib.use('Agg')  # <--ここを追加
from matplotlib import pyplot as plt
import os
import itertools

import networkx as nx
import datetime

import glob

#class iroiro_info():
#def __init__(self,dataset):
#    self.dataset=dataset

f_inf = np.inf

class network_test():
    def __init__(self,node_name,edge_con,edge_w) :
        #self.a=a        
        self.G=0
        self.node_name=node_name
        self.edge_con=edge_con
        self.edge_w=edge_w

    def test(G):
        nx.draw(G)

    def set(node_name,edge_con):
        Gh=nx.Graph()
        for i_node in range(len(node_name)):
            Gh.add_node(node_name[i_node])
        for i_edge in range(len(edge_con)):
            Gh.add_edge(edge_con[i_edge][0],edge_con[i_edge][1],weight=1/edge_con[i_edge][2])
        nx.draw(Gh, with_labels=True)
        plt.show()

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
    temp=0
    num_sum=0
    num=[]
    for i1 in range(i_rt):

        mat_x = exist_mat[i1][0].shape[1]
        mat_y = exist_mat[i1][0].shape[0]
        for y in range(mat_y):
            for x in range(mat_x):
                if(exist_mat[i1][0][y,x]==1):
                    temp=temp+1
        num.append(temp)
        num_sum=num_sum+temp
    return(num_sum,num)

def adj_node_tf(n,max):
    if 0<n<max:
        adj_range=np.array([-1,0,1])
    elif n==0:
        adj_range=np.array([0,1])
    else:
        adj_range=np.array([-1,0])
    return adj_range
        

def make_dist_mat(unit_h,unit_v,*node_mat):
    #node_mat: ノードTFを持った行列
    #unit_h:    水平ノード距離
    #unit_v:    垂直ノード距離
    if type(node_mat)==tuple:
        i_rt=len(node_mat)
    else:
        i_rt=1
        node_mat=np.array([node_mat])
    unit_diag = np.sqrt(unit_h**2+unit_v**2)    #斜め移動の距離
    mat_all=0
    print("...start making distance matrix...")
    rt_mat=[]
    for n in range(i_rt):
        mat_x = node_mat[n][0].shape[1]   #行列の横方向要素数
        mat_y = node_mat[n][0].shape[0]   #行列の縦方向要素数
        mat_all = (mat_x)*(mat_y) + mat_all
        mat_temp=np.identity(mat_y*mat_x)   #(x*y)次元の単位行列を生成
        #mat_temp=[]
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
                mat_all=mat_all-1

        rt_mat.append(con_mat)
    if i_rt==1:
        rt_data=warshal_floyd(mat_all,*rt_mat)
    else:
        rt_data=dist_mat_connect(mat_all,*rt_mat)
    return(rt_data)

def connecter(dist,seed=0,*dist_child_n):
    connect_points=get_train_dataset('connection',seed)
    for i in range(len(connect_points)):
        print(dist)


def dist_mat_connect(mat_all,*dist_mat):
    A = len(dist_mat)  #引数の数    
    dist_mat_size=np.empty(A,dtype=np.uint16)
    dist_mat_size_sum = mat_all
    
    print("...combining distance matrixes...")
    for i in range(A):
        dist_mat_size[i]=dist_mat[i].shape[0]  #i番目のdist_matのサイズ取得
        #dist_mat_size_sum = dist_mat_size_sum + dist_mat_size[i]
        #mat_zero[i]=np.full((dist_mat[i].shape[0],dist_mat[i].shape[0]),np.inf)
    mat_zero=[]
    
    for i2 in range(A):
        mat_zero.append([])
        for j2 in range(A):
            if(i2!=j2):
                mat_zero[i2].append(np.full((dist_mat_size[i2],dist_mat_size[j2]),f_inf))
            else:
                mat_zero[i2].append(dist_mat[i2])

    for i3 in range(A):
        mat_rt_htemp = mat_zero[i3][0]
        for j3 in range(1,A):
            mat_rt_htemp=np.hstack([mat_rt_htemp,mat_zero[i3][j3]])
        con_mat=np.vstack([con_mat,mat_rt_htemp]) if i3!=0 else mat_rt_htemp
    
    connection = get_train_dataset('connection')[0]
    weight = get_train_dataset('connection')[1]
    con_n = len(connection)
    if con_n!=len(weight):
        print()
        weight=list(weight[0] * con_n)
        print('!!!CAUTOIN!!!')
        print('conection weight list is shortage.')
    for i4 in range(con_n):
        #for j4 in range(connection[i4])
        con_mat[connection[i4][0],connection[i4][1]+dist_mat_size[0]]=weight[i4]
        con_mat[connection[i4][1]+dist_mat_size[0],connection[i4][0]]=weight[i4]    
    #con_mat[connection[0][0],connection[0][1]+dist_mat_size[0]]=0
    #con_mat[connection[0][1]+dist_mat_size[0],connection[0][0]]=0
    #con_mat[connection[1][0],connection[1][1]+dist_mat_size[0]]=0
    #con_mat[connection[1][1]+dist_mat_size[0],connection[1][0]]=0

    return(warshal_floyd(dist_mat_size_sum,con_mat))
    
def warshal_floyd(mat_all,con_mat):
    con_mat=np.array(con_mat)
    print("...start  to calculate time distance...")
    for k in range(mat_all):
        for i in range(mat_all):
            for j in range(mat_all):
                con_mat[i,j] = min(con_mat[i,j],con_mat[i,k]+con_mat[k,j])
    for l in range(mat_all-1,0,-1):
        if con_mat[0,l]==f_inf:
            con_mat=np.delete(np.delete(con_mat,l,0),l,1)
    dist_out(mat_all,con_mat)    
    return(con_mat,0)

def dist_out(mat_all,con_mat):
    n = np.arange(mat_all)
    out_mat = np.vstack([n,con_mat])
    n = np.insert(n,0,-1).reshape(-1,1)
    out_mat = np.hstack([n,out_mat])
    output_dir=os.path.join(dir_processing_file(),'dist_mat.csv')

    np.savetxt(output_dir,out_mat,fmt='%.2f',delimiter='\t')



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
    elif mode=='conection': #接続された点
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
                [1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,0,0,1,1,0,0,1,1],
                [1,1,1,0,0,1,1,0,0,1,1],
                [1,1,1,0,0,1,1,0,0,1,1],
                [0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0]
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
            rt_data=[[36,36],[5,5],[42,42]]
            rt2=[[1.2]]
    return (rt_data,rt2)

def exsit_2_number(mode,seed=0):
    h=1
    v=1
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
    if(mode=='pos'):
        #if len(exs)==1:
        #    rt_data=convert_exist2pos(h,v,exs)
        #    #rt_data=get_train_dataset('node_pos')
        #else:
        rt_data=convert_exist2pos(h,v,*exs)
    elif (mode=='dis'):
        #if len(exs)==1:
        #    rt_data=make_dist_mat(h,v,exs)
        #    #rt_data=get_train_dataset('distance')
        #else:
        rt_data=make_dist_mat(h,v,*exs)
    return(rt_data)

def file_import():
    path = os.getcwd()
    with open(path) as f:
        print(f.read)
    return(0)

def rt_dir(def_dir):
    #def_dir=os.path.join(os.path.dirname(__file__),"outputs")
    file_nth=dir_count(def_dir)+1
    #output_dir = datetime.datetime.now().strftime("%Y%m%d_%H_%M_")
    output_dir = 'test{:0=3}__processing'.format(file_nth)
    output_dir = os.path.join(def_dir, output_dir)
    return(output_dir)

def dir_count(out_dir):
    file_list=glob.glob(os.path.join(out_dir,'**/'))
    test_file_n=0
    for i in range(len(file_list)):
        file_list[i]=file_list[i].removeprefix(out_dir+'\\')
        if file_list[i].startswith("test"):
            test_file_n = test_file_n + 1
    #print(file_list)
    return(test_file_n)

def dir_processing_file(n=0):
    def_dir=os.path.join(os.path.dirname(__file__),"outputs")
    file_n=n if n!=0 else dir_count(def_dir)
    file_str="test{:0=3}__processing".format(file_n)
    return(os.path.join(def_dir,file_str))
        
   
def memo(path,txt):
    
    return()

def draw_map(data,labels,show=True,save=False,seed_num=0):
    x=data[:,0]
    y=data[:,1]
    fig,ax1=plt.subplots(1,1)
    #ax1.plot(self.x,self.y,s=50,c='b')
    #for i in range(len(labels)):
    #    l = labels[i]
    #    ax1.plot(x[i],y[i],'o',ms=10,mfc='b',mew=0)
    #    ax1.annotate(l,xy=(x[i],y[i]),size=15,color='r')
    
    for i in range(labels):
        ax1.plot(x[i],y[i],'o',ms=10,mfc='b',mew=0)
        ax1.annotate(i,xy=(x[i],y[i]),size=15,color='r')
        # plt.show()
    plt.axis('equal')
    #fig.canvas.mpl_connect('pick_event',onClick)
    if show:
        plt.show()
    if save:
        dir=dir_processing_file()
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
    print(exsit_2_number('pos',[0,1])[1])
    print(exsit_2_number('dis',[0,1]))