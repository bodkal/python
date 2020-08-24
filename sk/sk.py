import tkinter as tk 
import os
from functools import partial
import random

class card:
    def __init__(self,bord,id,x,y,color,ph):
        self.x=x
        self.y=y
        self.bk =  tk.PhotoImage(file=("./bk/{}.png".format(color)))
        self.image=tk.PhotoImage(file=r"./image/{}".format(ph)).subsample(2, 2) 
        self.Button=tk.Button(bord ,width=200,height=100, command=partial(change_color, self) )
        self.Button['image'] = self.image
        self.Button.place(x=self.x, y=self.y)
  
def change_color(card):
    card.Button['image'] = card.bk

def new_window1(Buttons):
    win1 = tk.Toplevel()
    win1.geometry("200x0") 
    revel=[]
    for i in Buttons:
        revel.append(tk.Button(win1, width=200,height=100))
        revel[-1]['image'] = i.bk
        revel[-1].place(x=i.x, y=i.y)
while 1:
    safe_closing=False
    Buttons=[]
    color=["b","b","b","b","b","b","b","b","r","r","r","r","r","r","r","r","r","bl","y","y","y","y","y","y","y"]
    img=os.listdir("./image/") 

    bord = tk.Tk() 
    bord.configure(background='gray')
    bord.geometry("1500x1100") 
    
    menubar = tk.Menu(bord)
    menubar.add_cascade(label="start over",command=bord.destroy)
    menubar.add_cascade(label="map",command=partial(new_window1,Buttons))# menu=filemenu)
    bord.config(menu=menubar)
    
    for j in range(5):
        for i in range(5):
            num=random.randint(0,len(color)-1)
            numimg=random.randint(0,len(img)-1)
            Buttons.append(card(bord,(i+j*5),50+i*250,20+j*140,color[num],img[numimg]))
            del color[num]
            del img[numimg]
    bord.bind('<Escape>', lambda e: exit())
    bord.mainloop() 


