import tkinter as tk

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from test_dataset import draw_map

import numpy as np

import convert_dataset as con_d
from convert_dataset import exsit_2_number


class map_events():
    def __init__(self,event=None):
        self.event=event
        self.gco = event.artist
        self.c_g=self.gco.get_color()
        self.l_g=self.gco.get_label()
        self.fig=plt.figure()
        
    
    def click_pt2(self):
        if (self.c == 'b'):
            self.gco.set_color('g')
        else: #(self.c=='g'):
            self.gco.set_color('b')
        self.fig.canvas.draw()




class map_viwer():
    def __init__(self,data,labels):
        self.x = data[:,0]
        self.y = data[:,1]
        self.labels=labels

        #self.fig = plt.figure()
        

    def onClick(event):
        ga=event.artist
        gx=ga.get_xdata()
        gy=ga.get_ydata()
        gi=event.ind
        ax1.plot(gx[gi],gy[gi],'o',ms=10,mfc='g',mew=0)

        #ax1.set_facecolor(color)
        plt.draw()

    def draw_map(self):
        
        fig,ax1=plt.subplots(1,1)
        #ax1.plot(self.x,self.y,s=50,c='b')
        for i in range(len(self.labels)):
            l = self.labels[i]
            ax1.plot(self.x[i],self.y[i],'o',ms=10,mfc='b',mew=0)
            ax1.annotate(l,xy=(self.x[i],self.y[i]),size=15,color='r')
            # plt.show()
        plt.axis('equal')
        #fig.canvas.mpl_connect('pick_event',onClick)
        plt.show()
        
    

class mainGUI():
    def __init__(self):
        self.root=tk.Tk()
        self.root.title("title")

        self.label1=tk.Label(self.root,text="hogehoge")


        self.label1.grid()
        self.root.mainloop()


if __name__ == "__main__":
    #mainGUI()
    node=exsit_2_number('pos')
    node_pos = node[0]
    node_num = node[1]
    map_data=map_viwer(node[0],range(node[1]))
    map_viwer.draw_map(map_data)