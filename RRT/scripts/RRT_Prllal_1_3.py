#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from treelib import Tree#, Node
import time
import os
from multiprocessing import Process, Value, Manager
from kinamtic_1_2 import kin_fun
from ROS_RUN import ROS_master as ros

import torch as th


class RRT:

    def __init__(self, current, desir, step_size, kenamatic, world_size, obstacle=0):
        self.jobs = []
        self.size = world_size
        # Desired start and end point
        self.current = current
        self.desir = desir
        # A matrix that holds all dots from both starts
        # The trees that arrange the matrix
        self.tree = Tree()
       

        # Did we finish the search
        self.NOF = Value("i", 0)
        self.statos = Value("i", 0)

        # self.winindex = Value("i", 0)
        self.winindex = Manager().list()
        self.badindex = Manager().list()
        self.badpoint = Manager().list()
        self.pool= (np.zeros((1,12)))
        self.pool_index=np.zeros(1)
        self.num_pool=os.cpu_count()-1
        # The number of dots already inserted into the tree
        self.NOP = 1

        # Step size
        self.step = step_size
        # Number of dimensions
        # self.NOD = len(current.T)
        self.t = time.time()
        self.obstacle = obstacle
        # The kinematics of the arms
        self.kin = kenamatic
        # Initial position of the arm base
        print("open new RRT model")

    def goliniar(self, point_a, point_b):
        mid_all_point = self.midpoint(point_a[0], point_b[0])
        i = 0
        dis = 1
        loops = len(mid_all_point)
        #print((self.obstacle_gpu.dtype),"68")
        while i < loops:
            ans = self.kin.configure_check_gpu(mid_all_point[i].reshape(1,12), self.obstacle_gpu)

            if ans==0:
                return i - dis, mid_all_point
            else:
                dis = max(int(ans**2 /4000), 1)
                i = i + dis
        return loops - 1, mid_all_point



       
    def goliniarpros(self, pointstart, pointend, index):

        print("tread {} start".format(index))
        secsed = True
        allpoints = self.midpoint(pointstart, pointend)
        i = 0
        loops = len(allpoints)

        while i < loops:
                
            ans = self.kin.configure_check(allpoints[i].reshape(1,12), self.obstacle)
            if not ans:
                secsed = False
                if(i>8):
                    self.badpoint.append(allpoints[i-8])
                    self.badindex.append(index)
                break
            else:
                i = i + max(int(ans**2 /3000), 1)
            if(self.statos.value):
                secsed = False
                break
            
        if secsed:
            self.statos.value = 1
            self.winindex.append(index)
            print("---tread {} secsed ---".format(index))
        #print((allpoints.dtype),"107")

        print("tread {} die ".format(index))
        self.NOF.value -= 1 
        

    
        def get_direction(self, tree, index):
            tree.show()
            #allrote = self.current
            if not index == 0:
                root = tree[index].predecessor(tree._identifier)
                good_route = tree[index].data
                while (not root == 0 or root == None):
                    point = np.array(tree[root].data)
                    good_route = np.append(point, good_route, axis=0)
                    root = tree[root].predecessor(tree._identifier)
                
                allrote = self.midpoint(self.desir[0],good_route[-1])
                allrote = np.append(allrote[:-1], self.midpoint(allrote[-1], good_route[0]), axis=0)
                for i in range(1, len(good_route)):
                    allrote = np.append(allrote[:-1], self.midpoint(good_route[i - 1], good_route[i]), axis=0)
                    
                allrote=np.append(allrote[:-1],self.midpoint(allrote[-1], self.current[0]), axis=0)
                
            else:
                allrote = self.midpoint(self.desir[0], self.current[0])
    # =============================================================================
    #             good_route = np.append(self.current, self.desir, axis=0)
    #             for i in range(1, len(good_route)):
    #                 allrote = np.append(allrote, self.midpoint(good_route[i - 1], good_route[i]), axis=0)
    # =============================================================================
               # allrote = allrote[1:, :]
            return np.flip(allrote,axis=0)

    # Extracting the path we found from the tree
    def get_direction1(self, tree, index):
        #tree.show()
        allrote = self.current
        if not index == 0:
            root = tree[index].predecessor(tree._identifier)
            good_route = tree[index].data
            while (not root == 0 or root == None):
                point = np.array(tree[root].data)
                good_route = np.append(point, good_route, axis=0)
                root = tree[root].predecessor(tree._identifier)
                
            for i in range(1, len(good_route)):
                allrote = np.append(allrote[:-1], self.midpoint(good_route[i - 1], good_route[i]), axis=0)
            allrote = np.append(allrote[1:], self.midpoint(self.all_point[index], self.desir[0]), axis=0)
            allrote = np.append(self.midpoint(allrote[0], allrote[1]), allrote[1:, :], axis=0)
            allrote = np.append(self.midpoint(self.current[0], allrote[0]),allrote, axis=0)

        else:
            good_route = np.append(self.current, self.desir, axis=0)
            for i in range(1, len(good_route)):
                allrote = np.append(allrote, self.midpoint(good_route[i - 1], good_route[i]), axis=0)
        return allrote

    def get_direction2(self, tree, index):
        #tree.show()
        allrote = self.current
        if not index == 0:
            root = tree[index].predecessor(tree._identifier)
            good_route = tree[index].data
            while (not root == 0 or root == None):
                point = np.array(tree[root].data)
                good_route = np.append(point, good_route, axis=0)
                root = tree[root].predecessor(tree._identifier)
                
            for i in range(1, len(good_route)):
                allrote = np.append(allrote,self.all_point[i].reshape(1,12), axis=0)
            allrote = np.append(allrote,self.desir, axis=0)

        print(tree.depth(index))
        return allrote

    # Gets a vector and normalizes it to a desired step size
    def get_normelize_vec(self, vec):
        return  ((vec / np.sqrt(np.sum(pow(vec,2) ))) *self.step)


    def midpoint(self, startp, endp):
        return np.linspace(startp, endp, int(np.max(np.abs(startp - endp) + 1)))

    # Gets a vector point and returns the geometrically closest vector point
    def min_distance(self, newpoint):
       return np.argmin(np.sum(pow(np.subtract(newpoint, self.all_point),2), axis=1))
   
    def dis_2point(self, point_A,point_B):
       return np.min(np.sqrt(np.sum(pow(np.subtract(point_A, point_B),2))))
   
    # Adds a point into the tree
    def add_to_tree(self, index, newpoint):
        self.tree.create_node(self.NOP, self.NOP, parent=index, data=newpoint)
        self.all_point = np.append(self.all_point, newpoint, axis=0)
        self.NOP += 1
        print("NOP : ", self.NOP)

    def bildwold(self):
        NOO = 200
        a = 1
        p = 50
        obs = np.array([np.ones(p) * NOO, np.linspace(-NOO, NOO, num=p), np.ones(p)]).T
        for i in range(1, int(NOO * a)):
            obs = np.append(obs, np.array([np.ones(p) * NOO, np.linspace(-NOO, NOO, num=p), np.ones(p) * (i)]).T,
                            axis=0)
            obs = np.append(obs, np.array([np.ones(p) * -NOO, np.linspace(-NOO, NOO, num=p), np.ones(p) * (i)]).T,
                            axis=0)

        obs = np.append(obs, np.array([np.linspace(-NOO, NOO, num=p), np.ones(p) * NOO, np.ones(p)]).T, axis=0)
        for i in range(1, int(NOO * a)):
            obs = np.append(obs, np.array([np.linspace(-NOO, NOO, num=p), np.ones(p) * NOO, np.ones(p) * (i)]).T,
                            axis=0)
            obs = np.append(obs, np.array([np.linspace(-NOO, NOO, num=p), np.ones(p) * -NOO, np.ones(p) * (i)]).T,
                            axis=0)

        NOO = NOO * 3
        for i in range(1, int(NOO * 2)):
            obs = np.append(obs, np.array([np.linspace(-NOO, NOO, num=p), np.ones(p) * (NOO - i), np.zeros(p)]).T,
                            axis=0)
        return (obs)

    def open_prosses(self, srart_pos,end_pos,index):
        p = Process(target=self.goliniarpros, args=(srart_pos, end_pos, index,))
        self.jobs.append(p)
        p.start()
        self.NOF.value += 1

    # Adds a point in a random direction
    def add_point(self, newpoint):

        index = self.min_distance(newpoint) #V
        
        newpoint = self.get_normelize_vec(np.subtract(newpoint, self.all_point[index])) + self.all_point[index]
        godindex, temppoints = self.goliniar(self.all_point[index].reshape(1,12), newpoint)
        #print((temppoints.dtype),"240")
        #print((self.all_point.dtype),"241")

        if (godindex > 0):
            self.add_to_tree(index,temppoints[godindex].reshape(1,12))
 
            if(self.NOF.value<self.num_pool):
                self.open_prosses(self.all_point[-1],self.desir[0],self.NOP - 1)
                                
            else:
                self.pool=np.append( self.pool,self.all_point[-1].reshape(1,12),axis=0)
                self.pool_index =np.append( self.pool_index,(self.NOP - 1))

            print("number of processes that are currently running : {}".format(self.NOF.value))

    def midserce1(self,allpoint):
        e=1
        max= int(len(allpoint) / 2)
        
        min=0
        mid = max
        new_Peza=int(max/4)
        old_peza=0
        while(abs(new_Peza-old_peza)>e):
            index_a = mid + new_Peza
            index_b = mid - new_Peza
            good_index, temp_points =self.goliniar(allpoint[index_a].reshape(1,12),allpoint[index_b].reshape(1,12))
            if (good_index+1==len(temp_points)):
                min=new_Peza
                old_peza=new_Peza
                new_Peza=int(new_Peza+(max-new_Peza)/2)
            else:
                max=new_Peza
                old_peza=new_Peza
                new_Peza=int(new_Peza-(new_Peza-min)/2 )
                #print("Peza = ",new_Peza)
        return new_Peza,mid


    def midserce(self,allpoint,max_val,min_val,start_point):
        #print("strt : {}\tmax : {}\tmin : {}".format(start_point,max_val,min_val))
        e=2
        
        new_index=int((max_val+min_val)/2)
        #print("new_index = ",new_index)

        old_index=0
        
        while(abs(new_index-old_index)>e):
            good_index, temp_points =self.goliniar(allpoint[start_point].reshape(1,12),allpoint[new_index].reshape(1,12))
            if (good_index+1==len(temp_points)):
                min_val=new_index
                old_index=new_index
                new_index=int(new_index+(max_val-new_index)/2)
            else:
                max_val=new_index
                old_index=new_index
                new_index=int(new_index-(new_index-min_val)/2 )
                if(new_index<min_val):
                    print("faild")
                    return old_index,False

        #print("end : ",new_index)
        return new_index,True



    def improved_path1(self, allpoint):
         print("\nsecend")
         t=time.time()
         a=len(allpoint)
         max_val=len(allpoint)
         min_val=0
         start_point=[0]
         new_beast=0
         old_beast=-10
         while max_val-new_beast>5 and new_beast- old_beast>2:
             old_beast=new_beast
             new_beast,stat=self.midserce(allpoint,max_val,min_val,start_point[-1])
             start_point.append(new_beast)
             min_val=new_beast
        
         
         optmaize=self.current
         j=0

         for i in start_point:
             optmaize=np.append(optmaize,self.midpoint(allpoint[j],allpoint[i]),axis=0)
             j=i
         if not stat:
             optmaize=np.append(optmaize,allpoint[start_point[-1]:,:],axis=0)
         else:
             optmaize=np.append(optmaize,self.midpoint(optmaize[-1],self.desir[0]),axis=0)

         allpoint = optmaize
         max_val=len(allpoint)

         print(-t+time.time())
         return self.improved_path(allpoint)
# =============================================================================
# 
# 
#     def improved_path2(self, allpoint):
#          print("\n\n\nstart -> 6")
#          t=time.time()
#          a=len(allpoint)
#          print("a->",a)
# 
#          max_val=len(allpoint)
#          min_val=0
#          start_point=[0]
#          new_beast=0
#          stat=True
#          dellist=[]
#          while max_val-new_beast>1 :
#              old_beast=new_beast
#              new_beast,stat=self.midserce1(allpoint,max_val,min_val,start_point[-1])
#              start_point.append(new_beast)
#              min_val=new_beast
#         
#          print(start_point,"-----------")
#          if len(start_point)>1:
#              for i in range(1,len(start_point)):      
#                  for i in (range(start_point[i-1]+1,start_point[i])):
#                      dellist.append(i)
#              print(dellist)
#              b=np.delete(allpoint,dellist,0)
#          print("b->",len(b))
# 
#          optmaize=self.current        
#          for i in range(1,len(b)):
#              optmaize=np.append(optmaize,self.midpoint(b[i-1],b[i]),axis=0)
#         
#          allpoint = optmaize
#          for i in range(1,len(allpoint)):
#              if (not (np.linalg.norm(allpoint[i-1]-allpoint[i]))):
#                  print(0)
#          max_val=len(allpoint)
#          print("old : {} \t new : {}".format(a,len(allpoint)))
#          print(-t+time.time())
#          return allpoint
# =============================================================================
     


    def improved_path(self, allpoint):
         print("first")
         t=time.time()

         a=len(allpoint)
         beast,mid=self.midserce1(allpoint)        
            
         if (not beast == 0):
             tempend   = allpoint[ (mid + beast-1):,:]
             tempstart = allpoint[:(mid - beast+1),:]
             if(not len(tempend)):
                 tempend=allpoint[-1,:].reshape((1,12))
             if(not len(tempstart)):
                 tempend=allpoint[0,:].reshape((1,12))
             tempmid = self.midpoint(tempstart[-1,: ], tempend[0,: ])
             temp = np.append(tempstart, tempmid[1:-1], axis=0)
             allpoint = np.append(temp, tempend, axis=0)
                
         #print("old : {} \t new : {}".format(a,len(allpoint)))
         print(-t+time.time())
         return allpoint
    
    

    def get_winindex(self):
        print(self.winindex)
        print("Extracting the route")
        dis = self.tree.depth(self.winindex[0]) * self.step + np.linalg.norm(self.tree[self.winindex[0]].data - self.desir)
        min_val = 99999
        for i in self.winindex:
            dis = self.tree.depth(i) * self.step + np.linalg.norm(self.tree[i].data - self.desir)
            print(i, "\t", self.tree.depth(i), "\t", np.linalg.norm(self.tree[i].data - self.desir), '\t', dis)

            if (dis < min_val):
                min_val = dis
                win = i
        print(win)
        self.winindex = Manager().list()
        return win

    # The function that activates everything
    def let_the_magic_begin(self, ros_fun,isflipt):

        self.open_prosses(self.current[0],self.desir[0],self.NOP - 1)
        NOL=0
        while (not self.statos.value):
            if (not NOL % 10):
                self.size += 10
                self.size=min(self.size,360)
                
            newpoint =  ((np.random.rand(1, 12,)) * self.kin.limit * self.size - self.kin.fas)
            #print((newpoint.dtype),"436")
            self.add_point(newpoint)
            
            if(len(self.pool)>1):
                while(len(self.pool)>1 and self.NOF.value<self.num_pool):
                    if(self.statos.value):
                        break
                    self.open_prosses(self.pool[-1],self.desir[0],int(self.pool_index[-1]))
                    self.pool=np.delete(self.pool,-1,0)
                    self.pool_index=np.delete(self.pool_index,-1)
           
            if(len(self.badindex)):
                for i in range(len(self.badindex)):
                    self.add_to_tree(self.badindex[0],self.badpoint[0].reshape(1,12))
                    del self.badindex[0]
                    del self.badpoint[0]
            NOL+=1
            
                    

        print("Waiting for all processes to die")
        for job in self.jobs:
            job.join()
        best_point_index = self.get_winindex()
        allrote = self.get_direction1(self.tree, best_point_index)
        
        if isflipt:
           allrote=np.flip(allrote,axis=0)
           tmp=self.desir
           self.desir=self.current
           self.current=tmp
         
        print(time.time() - self.t)
        input("preace any key to improved path\n")
        if(best_point_index==0):
            print("no need to improve path")
            allrote1=allrote
        #else:
            
# =============================================================================
#             for i in allrote:
#                 print(self.kin.configure_check1(i.reshape(1,12),self.obstacle_gpu))
#                 if not (self.kin.configure_check1(i.reshape(1,12),self.obstacle_gpu)):
#                     print("faild")
# =============================================================================
# =============================================================================
# 
#         allrotetemp = self.get_direction2(self.tree, best_point_index)
#         allrote2=self.improved_path2(allrotetemp)
#         print("first : old -> {} \tnew -> {}".format(len(allrote),len(allrote2)))
#         allrote3=self.improved_path1(allrote2)
#         print("sec : old -> {} \tnew -> {}".format(len(allrote),len(allrote2)))
# =============================================================================
        file="gal1"
        with open(file,'wb') as f: pickle.dump(allrote, f)
        
        allrote6=self.improved_path1(allrote)
        print("fird : old -> {} \tnew -> {}".format(len(allrote),len(allrote6)))

        #allrote6=self.improved_path(allrote6)
        #allrote3=self.improved_path(allrote)
        #plotallpach(allrote6,allrote)
        file="gal2"
        with open(file,'wb') as f: pickle.dump(allrote6, f)

 #       with open(file,'rb') as f: arrayname1 = pickle.load(f)

 #       print(np.array_equal(allrote6,arrayname1))
        
        input("preace any key to run the simulation\n")
        ros_fun.send_to_arm(allrote6 * np.pi / 180)
        #input("preace any key to run the simulation\n")
        #ros_fun.send_to_arm(np.flip(allrote,axis=0) * np.pi / 180)


    def run_serce(self, ros_fun, isinvers, invers_pos):


        self.obstacle_gpu=th.from_numpy(self.obstacle).float().to('cuda')
        
        if (isinvers):
            self.desir = self.kin.invers_arms_cfg(invers_pos)
        self.t = time.time()
        
        dis_des=self.kin.configure_check_gpu(self.desir, self.obstacle_gpu)
        dis_crr=self.kin.configure_check_gpu(self.current, self.obstacle_gpu)
        
        if dis_des:
            if dis_des<dis_crr:
                temp=self.desir
                self.desir=self.current
                self.current=temp
                isflipt=True
                print("desir = {} \ncurrent={} \nneed to flip".format(dis_des, dis_crr))
            else:
                isflipt=False
                print("desir = {} \ncurrent={} \nno need to flip".format(dis_des, dis_crr))

                
            self.tree.create_node(0, 0, data=self.current)  # root node
            self.all_point = self.current
            #print(self.current.dtype)
            self.let_the_magic_begin(ros_fun,isflipt)
        else:
            print("can't go to disre location")
            
        # model.print_tree_nd(start)
def plotallpach(allpointopt,allpoint):
    size=len(allpoint)/len(allpointopt)
    for i in range(len(allpoint[0])):
        plt.subplot(len(allpointopt[0]),1,i+1)
        plt.plot(range(len(allpoint)),allpoint[:,i], 'b')
    
    for i in range(len(allpointopt[0])):
        plt.subplot(len(allpointopt[0]),1,i+1)
        plt.plot(np.array(range(len(allpointopt)))*size,allpointopt[:,i], 'r')
    plt.show()

if __name__ == '__main__':
    kin = kin_fun('../include/ManipulatorH')
    # size = 360
    size = 90
    
    # Large step size software executes
    step = 12 * 2
    
    end = np.array([[0, 0, -90, 0, 0, 0, 0, 0, -90, 0, 0, 0]])

    
    #pos = np.array([np.random.random_sample(1)*400, np.random.random_sample(1)*700-350,np.random.random_sample(1)*300+100,np.random.random_sample(1)*-400, np.random.random_sample(1)*700-350, np.random.random_sample(1)*300+100])
    #ori =  np.random.random_sample((1,6))*1.6-0.8
    #(arm,-arm)
    pos = np.array([100,300, 100,-100,-300, 100])
    ori = np.array([ 0, np.pi/2, 0, 0,  np.pi/2, 0])
    print("pos ->",pos)
    print("ori ->",ori)

    #ori = np.array([0, 0, 0, 0, 0, 0])

    ros_meneger = ros(100)
    start = (ros_meneger.readstat())
    
    model = RRT(start, end, step, kin, size, 0)
    model.obstacle = model.bildwold()
    model.run_serce(ros_meneger, isinvers=True, invers_pos=np.append(pos,ori))
    


#TO DO
#line in invers kinmtic

#dun
#cainge start -> end  V
#fix index pross V
#fix impruve 2
#improve improve pach V
#try index finis first V
# trnsfer to flaut32 X

