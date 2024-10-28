import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
import datetime

#class iroiro_info():
#def __init__(self,dataset):
#    self.dataset=dataset

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

def convert_exist2pos(exist_mat,unit_h,unit_v):
    mat_x = exist_mat.shape[1]
    mat_y = exist_mat.shape[0]
    pos_mat=np.arange(2).reshape(1,2)
    i = 0
    for y in range(mat_y):
        for x in range(mat_x):
            if(exist_mat[y,x]==1):
                posi = np.array([(-mat_x+x)*unit_h,(mat_y-y)*unit_v]).reshape(1,2)
                pos_mat=np.vstack((pos_mat,posi))
                i += 1
    #pos_mat=pos_mat.reshape(i,2)
    pos_mat=np.delete(pos_mat,0,axis=0)
    return(pos_mat,i)


def make_dist_mat(node_mat,unit_h,unit_v):
    f_inf=99999
    #node_mat: ノードTFを持った行列
    #unit_h:    水平ノード距離
    #unit_v:    垂直ノード距離
    unit_diag = np.sqrt(unit_h**2+unit_v**2)    #斜め移動の距離
    mat_x = node_mat[0].shape[1]   #行列の横方向要素数
    mat_y = node_mat[0].shape[0]   #行列の縦方向要素数
    mat_all = (mat_x)*(mat_y)
    mat_temp=np.identity(mat_y*mat_x)   #(x*y)次元の単位行列を生成
    con_mat = np.where(mat_temp==0,f_inf,0).astype(np.float32)
    for y in range(mat_y):  #n行目の処理
        if 0<y<mat_y-1:
            y_temp_range=[-1,0,1]
        elif y==0:
            y_temp_range=[0,1]
        else:
            y_temp_range=[-1,0]
        for x in range(mat_x):  #m列目の処理
            if 0<x<mat_x-1:
                x_temp_range=[-1,0,1]
            elif x==0:
                x_temp_range=[0,1]
            else:
                x_temp_range=[-1,0]
            if node_mat[0][y,x]!=0:
                for y_temp in y_temp_range:
                    for x_temp in x_temp_range:
                        if(node_mat[0][y+y_temp,x+x_temp]==1):
                            if(x_temp*y_temp!=0):
                                con_mat[(mat_x)*y+x,(mat_x)*(y+y_temp)+x+x_temp]=unit_diag
                            elif(x_temp==0 and y_temp==0):
                                con_mat[(mat_x)*y+x,(mat_x)*(y+y_temp)+x+x_temp]=0
                            elif(x_temp==0):
                                con_mat[(mat_x)*y+x,(mat_x)*(y+y_temp)+x+x_temp]=unit_v
                            elif(y_temp==0):
                                con_mat[(mat_x)*y+x,(mat_x)*(y+y_temp)+x+x_temp]=unit_h
    for k in range(mat_all):
        for i in range(mat_all):
            for j in range(mat_all):
                con_mat[i,j] = min(con_mat[i,j],con_mat[i,k]+con_mat[k,j])
    return(con_mat,0)

def get_train_dataset(mode):
    rt2=0
    if mode=='node_pos':  #各点の位置
        rt_data=np.array([
            [1,1],
            [2,2],
            [1,2],
            [2,1]
            ])
        rt2=rt_data.shape[0]
    elif mode=='distance':  #点同士の所要時間
        rt_data =np.array([
            [ 0, 4, 1, 1],
            [ 4, 0, 5, 1],
            [ 1, 5, 0, 2],
            [ 1, 1, 2, 0]
            ])
    elif mode=='conection': #接続された点
        rt_data = np.array([[1,2],
                            [0,3],
                            [0,3],
                            [1,2]])
    elif mode=='node_exist':
        rt_data = np.array([
            [1,1,1,1],
            [1,1,0,1],
            [0,0,1,1],
            [1,1,1,1]
            ]).astype(np.uint8)
        #9*9glid
    return (rt_data,rt2)

def exsit_2_number(mode):
    h=1
    v=1
    exs=get_train_dataset('node_exist')
    if(mode=='pos'):
        rt_data=convert_exist2pos(exs,h,v)
        #rt_data=get_train_dataset('node_pos')
    elif (mode=='dis'):
        rt_data=make_dist_mat(exs,h,v)
        #rt_data=get_train_dataset('distance')
    return(rt_data)


if __name__=="__main__":
    #a=np.array([
    #        [0,1,1,0],
    #        [1,0,0,1],
    #        [1,0,0,1],
    #        [0,1,1,0]
    #    ])
    #G = nx.Graph(a)
    unit_h=1
    unit_v=1
    test_nd = get_train_dataset('node_exist')
    print(exsit_2_number('dis'))
#    g_name_list=range(4)
#    g_conection=([(0,1,1),
#                  (1,2,1),
#                  (2,3,1),
#                  (3,4,1),
#                  (0,4,1),
#                  (1,4,10),
#                  #(4,0,1),
#                  (0,2,1)
#                  ])
#    network_test.set(g_name_list,g_conection)

#//------------------test--------------------//
#    g_name_list=["A","B","C","D"]
#    g_conection=([("A","B",1),
#                  ("A","C",1),
#                  ("A","D",2),
#                  ("B","C",2),
#                  ("B","D",1),
#                  ("C","D",1)])
#//-------------------------------------------//
#    g_name_list=["Fuchinobe","Kobuchi","Machida","Naruse","Nagatsuta","Sagami-Ono","Tama","Oda-Saga"]
#    g_conection=([("Fuchinobe","Kobuchi",3),
#                  ("Kobuchi","Machida",4),
#                  ("Machida","Naruse",3),
#                  ("Naruse","Nagatsuta",3),
#                  ("Machida","Sagami-Ono",1),
#                  ("Machida","Tama",2),
#                  ("Sagami-Ono","Oda-Saga",2),
#                  ("Kobuchi","Tama",2)])
#

#[[1,1],
#[2,2],
#[3,3],
#[4,4],
#[5,5],
#[3,5],
#[3,4],
#[4,3],
#[5,3],
#[5,2]]


#[[ 0, 1, 4, 6, 7, 8, 5, 6, 9,11],
#[ 1, 0, 3, 5, 6, 7, 4, 5, 8,10],
#[ 4, 3, 0, 2, 3, 4, 1, 2, 5, 7],
#[ 6, 5, 2, 0, 1, 6, 3, 4, 7, 9],
#[ 7, 6, 3, 1, 0, 7, 4, 5, 8,10],
#[ 8, 7, 4, 6, 7, 0, 3, 6, 9,11],
#[ 5, 4, 1, 3, 4, 3, 0, 3, 6, 8],
#[ 6, 5, 2, 4, 5, 6, 3, 0, 3, 5],
#[ 9, 8, 5, 7, 8, 9, 6, 3, 0, 2],
#[11,10, 7, 9,10,11, 8, 5, 2, 0]]