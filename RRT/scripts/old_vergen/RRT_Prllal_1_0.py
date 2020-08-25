#!/usr/bin/env python3

import numpy as np
# import matplotlib.pyplot as plt
from treelib import Tree#, Node
import time

from multiprocessing import Process, Value, Manager
from kinamtic_1_0 import rotation_matrices
from ROS_RUN import ROS_master as ros

import torch as th

# =============================================================================
# from torch.multiprocessing import Pool, set_start_method, Value, Manager
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass
# =============================================================================
# from os import path
# from scipy.spatial.distance import cdist


class RRT:

    def __init__(self, current, desir, step_size, kenamatic, world_size, obstacle=0):
        self.jobs = []
        self.size = world_size
        # Desired start and end point
        self.current = current
        self.desir = desir
        # A matrix that holds all dots from both starts
        self.all_point = self.current
        # The trees that arrange the matrix
        self.tree = Tree()
        self.tree.create_node(0, 0, data=self.current)  # root node

        # Did we finish the search
        self.NOF = Value("i", 0)
        self.statos = Value("i", 0)

        # self.winindex = Value("i", 0)
        self.winindex = Manager().list()
        self.badindex = Manager().list()
        self.badpoint = Manager().list()
        self.pool=np.zeros((1,12))
        self.num_pool=9
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
        #print('loop:',loops)
        while i < loops:
            ans = self.kin.configure_check1(mid_all_point[i].reshape(1,12), self.obstacle_gpu)
            if ans==0:
                return i - dis, mid_all_point
            else:
                dis = max(int(ans**2 /4000), 1)
                i = i + dis
                #print(ans,' -->\t',dis)
        return loops - 1, mid_all_point



        
    def goliniarpros(self, pointstart, pointend, index, winindex,badindex,badpoint):
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
                    badpoint.append(allpoints[i-8])
                    badindex.append(index)
                break
            else:
                g=max(int(ans**2 /3000), 1)
                i = i + g
                #print(ans,' -->\t',g)


        if secsed:
            self.statos.value = 1
            winindex.append(index)
            print("---tread {} secsed ---".format(index))

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
           # allrote = allrote[1:, :]
        return allrote

    # Gets a vector and normalizes it to a desired step size
    def get_normelize_vec(self, vec):
        vec_size = (np.sum(vec ** 2)) ** 0.5
        normelize_vec = (vec / vec_size) * self.step
        return normelize_vec

    def midpoint(self, startp, endp):
        return np.linspace(startp, endp, int(np.max(np.abs(startp - endp) + 1)))

    # Gets a vector point and returns the geometrically closest vector point
    def min_distance(self, newpoint, data_points):
        diff = np.subtract(newpoint, data_points)
        index = np.argmin(np.sum(diff[:, :] ** 2, axis=1))
        # distance = np.sum(diff[index] ** 2) ** 0.5
        return index

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
        return obs

    def open_prosses(self, srart_pos,end_pos,index):
        p = Process(target=self.goliniarpros, args=(srart_pos, end_pos, index, self.winindex,self.badindex,self.badpoint,))
        self.jobs.append(p)
        p.start()
        self.NOF.value += 1

    # Adds a point in a random direction
    def add_point(self, newpoint):

        index = self.min_distance(newpoint, self.all_point)
        
        newpoint = self.get_normelize_vec(np.subtract(newpoint, self.all_point[index])) + self.all_point[index]
        godindex, temppoints = self.goliniar(self.all_point[index].reshape(1,12), newpoint)

        if (godindex > 0):
            self.add_to_tree(index,temppoints[godindex].reshape(1,12))
            if(self.NOF.value<self.num_pool):
                self.open_prosses(self.all_point[-1],self.desir[0],self.NOP - 1)
            else:
                self.pool=np.append( self.pool,self.all_point[-1].reshape(1,12),axis=0)
            print("number of processes that are currently running : {}".format(self.NOF.value))

    def improved_path1(self, allpoint):
        print("1")
        t=time.time()
        templen=1000000
        while(templen-len(allpoint)>10):
            a=len(allpoint)
            mid = int(len(allpoint) / 2)
            Peza = 5
            index_a = mid + Peza
            index_b = mid - Peza
            beast = 0
            templen=len(allpoint)

            while (index_a <= len(allpoint)-1 and index_b >= 0):
                good_index, temp_points =self.goliniar(allpoint[index_a].reshape(1,12),allpoint[index_b].reshape(1,12))
                if (good_index+1==len(temp_points)):
                    beast = Peza
                Peza +=  5
                index_a =mid+ Peza
                index_b =mid- Peza
                
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
                
            print("old : {} \t new : {}".format(a,len(allpoint)))
        print(-t+time.time())
        return allpoint
    
    
    def improved_path2(self, allpoint):
        print("2")

        t=time.time()
        templen=1000000
        for i in range(1,4):
            mid=int(len(allpoint) /4*i )
            Peza = 5
            index_a = mid + Peza
            index_b = mid - Peza
            beast = 0
            templen=len(allpoint)

            while (index_a <= len(allpoint)-1 and index_b >= 0):
                good_index, temp_points =self.goliniar(allpoint[index_a].reshape(1,12),allpoint[index_b].reshape(1,12))
                if (good_index+1==len(temp_points)):
                    beast = Peza
                else:
                    break
                
                Peza +=  5
                index_a =mid+ Peza
                index_b =mid- Peza
                
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

# =============================================================================
#         i=len(allpoint)-5
#         while(i<0):    
#             good_index, temp_points =self.goliniar(allpoint[i].reshape(1,12),allpoint[-1].reshape(1,12))
#             if (good_index+1==len(temp_points)):
#                 beast = i
#             else:
#                 break
#             i -=  5
#         if (not beast == 0):
#             tempmid = self.midpoint( allpoint[beast-1],allpoint[-1])
#             allpoint = np.append(allpoint[(beast-1):],tempmid,  axis=0)
# =============================================================================

        print(-t+time.time())
        return allpoint
    
    def improved_path3(self, allpoint):
        print("4")

        t=time.time()
        templen=1000000
        a=len(allpoint)
        i=5
        a=len(allpoint)

        while(i<len(allpoint)):    
            good_index, temp_points =self.goliniar(allpoint[0].reshape(1,12),allpoint[i].reshape(1,12))
            if (good_index+1==len(temp_points)):
                beast = i
            else:
                break
            i +=  5
        if (not beast == 0):
            tempmid = self.midpoint(allpoint[0], allpoint[beast-1])
            allpoint = np.append(tempmid, allpoint[(beast-1):], axis=0)
                
        i=len(allpoint)-1
        
        print("old : {} \t new : {}".format(a,len(allpoint)))
        a=len(allpoint)

        while(i>0):    
            good_index, temp_points =self.goliniar(allpoint[i].reshape(1,12),allpoint[-1].reshape(1,12))
            if (good_index+1==len(temp_points)):
                beast = i
            else:
                break
            i -=  5
        if (not beast == len(allpoint)-1):
            tempmid = self.midpoint(allpoint[beast-1],allpoint[-1])
            allpoint = np.append(allpoint[:(beast-1)],tempmid,  axis=0)

            print("old : {} \t new : {}".format(a,len(allpoint)))
            a=len(allpoint)

            mid=int(len(allpoint) /2 )
            Peza = 5
            index_a = mid + Peza
            index_b = mid - Peza
            beast = 0
            templen=len(allpoint)

            while (index_a <= len(allpoint)-1 and index_b >= 0):
                good_index, temp_points =self.goliniar(allpoint[index_a].reshape(1,12),allpoint[index_b].reshape(1,12))
                if (good_index+1==len(temp_points)):
                    beast = Peza
                Peza +=  5
                index_a =mid+ Peza
                index_b =mid- Peza
                
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
                
            print("old : {} \t new : {}".format(a,len(allpoint)))

        print(-t+time.time())
        return allpoint
    
    
# =============================================================================
#     def improved_path(self, allpoint):
#         print("imp -> 0")
#         t=time.time()
#         templen=1000000
#         while(templen-len(allpoint)>10):
#             a=len(allpoint)
# 
#             mid = int(len(allpoint) / 2)
#             Peza = 4
#             index_a = mid + Peza
#             index_b = mid - Peza
#             templen=len(allpoint)
#             while (index_a <= len(allpoint)-1 and index_b >= 0):
#                 self.open_prosses(allpoint[index_a],allpoint[index_b],Peza)
#                 Peza +=  4
#                 index_a =mid+ Peza
#                 index_b =mid- Peza
#             print("Waiting for all processes to die")
#             for job in self.jobs:
#                 job.join()
#             beast=max(self.winindex)
#             if (not beast == 0):
#                 tempend   = allpoint[ (mid + beast-2):,:]
#                 tempstart = allpoint[:(mid - beast+2),:]
#                 if(not len(tempend)):
#                     tempend=allpoint[-1,:].reshape((1,12))
#                 if(not len(tempstart)):
#                     tempend=allpoint[0,:].reshape((1,12))
#                 tempmid = self.midpoint(tempstart[-1], tempend[0])
#                 temp = np.append(tempstart, tempmid[1:-1], axis=0)
#                 allpoint = np.append(temp, tempend, axis=0)
#             print("old : {} \t new : {}".format(a,len(allpoint)))
# 
#         print(-t+time.time())
#         for job in self.jobs:
#                 job.join()
#         return allpoint
# =============================================================================

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
    def let_the_magic_begin(self, ros_fun):

        self.open_prosses(self.current[0],self.desir[0],self.NOP - 1)
        NOL=0
        while (not self.statos.value):
            if (not NOL % 10):
                self.size += 10
                self.size=min(self.size,360)
                
            newpoint = (np.random.rand(1, 12) * self.kin.limit * self.size - self.kin.offset)
            self.add_point(newpoint)
            
            if(len(self.pool)>1):
                while(len(self.pool)>1 and self.NOF.value<self.num_pool):
                    self.open_prosses(self.pool[-1],self.desir[0],self.NOP - 1)
                    self.pool=np.delete(self.pool,-1,0)

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
        print(time.time() - self.t)
        input("preace any key to improved path\n")
        if(best_point_index==0):
            print("no need to improve path")
            allrote1=allrote
        else:
            
            for i in allrote:
                if not (self.kin.configure_check1(i.reshape(1,12),self.obstacle_gpu)):
                    print("faild")
                
            allrote2=self.improved_path1(allrote)
            for i in allrote2:
                if not (self.kin.configure_check1(i.reshape(1,12),self.obstacle_gpu)):
                    print("faild")

            allrote3=self.improved_path2(allrote)
            for i in allrote3:
                if not (self.kin.configure_check1(i.reshape(1,12),self.obstacle_gpu)):
                    print("faild")

            allrote4=self.improved_path3(allrote)
            for i in allrote4:
                if not (self.kin.configure_check1(i.reshape(1,12),self.obstacle_gpu)):
                    print(0)
                    print("faild")
            
           
          #  allrote2=self.improved_path(allrote)
            print("old -> : {}".format(len(allrote)))
            print("1 -> : {} imp -> {}".format(len(allrote2),len(allrote)-len(allrote2)))
            print("2 -> : {} imp -> {}".format(len(allrote3),len(allrote)-len(allrote3)))
            print("3 -> : {} imp -> {}".format(len(allrote4),len(allrote)-len(allrote4)))

        input("preace any key to run the simulation\n")
        ros_fun.send_to_arm(allrote2 * np.pi / 180)
        input("preace any key to run the simulation\n")
        ros_fun.send_to_arm(np.flip(allrote,axis=0) * np.pi / 180)


    def run_serce(self, ros_fun, isinvers, invers_pos):

        # The RRT builder gets a starting point,end point, step size, kinematics, 3-dimensional cloud of obstacles
        self.kin.configure_check(self.desir, self.obstacle)

        self.obstacle_gpu=th.from_numpy(self.obstacle).float().to('cuda')
        
        if (isinvers):
            self.desir = self.kin.invers_arms_cfg(invers_pos)
        self.t = time.time()

        if self.kin.configure_check1(self.desir, self.obstacle_gpu):
            
            self.let_the_magic_begin(ros_fun)
        else:
            print("can't go to disre location")

        # model.print_tree_nd(start)

if __name__=='__main__':
    #__spec__ = None
    kin = rotation_matrices('../include/ManipulatorH')
    # size = 360
    size = 90
    
    # Large step size software executes
    step = 12 * 2
    
    end = np.array([[0, 0, -90, 0, 0, 0, 0, 0, -90, 0, 0, 0]])
    #(arm,-arm)
    #pos = np.array([50, 100, 400, -50,100, 400])
    pos = np.array([100, 0, 100,-350,0, 100])
    #pos = np.array([211,125, 350,-112,0, 378])

    ori = np.array([ 0, np.pi/2, 0, 0, np.pi/2, 0])
    #ori = np.array([0, 0, 0, 0, 0, 0])

    ros_meneger = ros(100)
    start = ros_meneger.readstat()
    
    model = RRT(start, end, step, kin, size, 0)
    model.obstacle = model.bildwold()
    model.run_serce(ros_meneger, isinvers=True, invers_pos=np.append(pos,ori))
    
# https://dask.org/
#torch.multiprocessing
#https://towardsdatascience.com/speed-up-your-algorithms-part-3-parallelization-4d95c0888748


# TO  DO : path inverce rachef

    # t=time.time()
    # for i in range(1000):
    #     kin.configure_check1(end,model.obstacle)
    # print(time.time()-t)
    # t=time.time()
    # for i in range(1000):
    #     kin.configure_check(end,model.obstacle)
    # print(time.time()-t)
    # end=end+1
    # b=kin.configure_check1(end,0)
    # print(a[0,:])
    # print(b[0,:])
    # print(np.max(np.linalg.norm(a-b,axis=1)))
    

    # Printing the way for N dimensions gets the path we found
    # =============================================================================
    #     def print_tree_nd(self, good_route):
    #         #fig = plt.figure(0)
    #         # fig, axs = plt.subplots(len(self.current.T))
    #         # for i in range(len(self.current.T)):
    #         #     axs[i].plot(1, self.current[0, i])
    #         #     axs[i].plot(1, self.desir[0, i])
    #         #     temp_good_route = good_route.T[i]
    #         #     for j in range(len(good_route) - 1):
    #         #         axs[i].plot([j, j + 1], [temp_good_route[j], temp_good_route[j + 1]], 'k')
    #         # plt.show()
    #
    #         for i in good_route:
    #             a = self.kenamatic.forword(i[:6])
    #
    #             a = np.array([a[:, 0] * -1, a[:, 1] * -1, a[:, 2]]).T
    #             kenamatic.arm_plot(self.kenstart1 + a, 1, self.obstacle)
    #             kenamatic.arm_plot(self.kenstart2 + kenamatic.get_all_joint(i[-6:] * np.pi / 180.0), 1, self.obstacle)
    #             plt.show()
    # =============================================================================

    #            plt.draw()
    #            plt.pause(0.0001)
    #            plt.clf()

  
