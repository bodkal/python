#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,cos,pi
#import pickle
import sympy as sp
import yaml
from scipy.spatial.distance import cdist
import torch as th
import time

class rotation_matrices:
    def __init__(self, filename):

        print("loding data from {} ... ".format(filename+'.yaml'))

        self.filename=filename

        with open(self.filename+'.yaml', 'r') as ymlfile:
            data = yaml.load(ymlfile)
            
        self.rotate = data['kinmatic']['Rotnson'].split(',')
        self.angels = data['kinmatic']['angels'].split(',')
        self.move = np.array(data['kinmatic']['move'].split(','))

        self.length = np.array(data['arminf']['Length'].split(','),dtype=float)
        self.ofseet = np.array(data['arminf']['ofseet'].split(','),dtype=float)

        self.kenstart1 =np.array([data['armpos']['arm1'].split(',')])
        self.kenstart2 =np.array([data['armpos']['arm2'].split(',')])
        print("finise loding data")

    def get_all_joint(self, allteta):
        return np.array(self.all_joint.subs(list(zip(self.sym_t, allteta)))[:, :3])
    
    def multiplication(self, A, angel,len, vec, xyz):
        if (xyz[1] == 'x'):
            L = sp.Matrix([len, 0, 0])
        elif (xyz[1] == 'y'):
            L = sp.Matrix([0,len, 0])
        elif (xyz[1] == 'z'):
            L = sp.Matrix([0, 0, len])
        else:
            L = sp.Matrix([0, 0, 0])

        if (xyz[0] == 'x'):
            A = (A * self.Ax(angel,L))
        elif (xyz[0] == 'y'):
            A = (A * self.Ay(angel,L))
        elif (xyz[0] == 'z'):
            A = (A * self.Az(angel,L))
        elif (xyz[0] == 'd'):
            A = (A * self.Ad(angel))

        if (not xyz[1] == 'o'):
            vec = vec.row_insert(0, sp.trigsimp(A * self.zerovec).T)

        #return sp.trigsimp(A.subs(list(zip(self.sym_l, self.length)))), sp.trigsimp(vec.subs(list(zip(self.sym_l, self.length))))
        return sp.sympify(sp.trigsimp(A)), sp.sympify(sp.trigsimp(vec))

            
    def Ax(self, t,L = sp.Matrix([0, 0, 0])):
        return sp.Matrix([[1, 0, 0, L[0]],
                          [0, sp.cos(t), -sp.sin(t), L[1]],
                          [0, sp.sin(t), sp.cos(t), L[2]],
                          [0, 0, 0, 1]])

    def Ay(self, t, L = sp.Matrix([0, 0, 0])):
        return sp.Matrix([[sp.cos(t), 0, sp.sin(t), L[0]],
                          [0, 1, 0, L[1]],
                          [-sp.sin(t), 0, sp.cos(t), L[2]],
                          [0, 0, 0, 1]])

    def Az(self, t,L = sp.Matrix([0, 0, 0])):
        return sp.Matrix([[sp.cos(t), -sp.sin(t), 0, L[0]],
                          [sp.sin(t), sp.cos(t), 0, L[1]],
                          [0, 0, 1, L[2]],
                          [0, 0, 0, 1]])

    def Ad(self, L):
        return sp.Matrix([[1, 0, 0, L[0]],
                          [0, 1, 0, L[1]],
                          [0, 0, 1, L[2]],
                          [0, 0, 0, 1]])

    def get_all_sym_joint(self,b):
        print("start calculate kenematics ")
        A = self.A
        for i in range(b):#len(self.rotate)):
            A, self.all_joint = self.multiplication(A, self.angels[i],self.move[i], self.all_joint, self.rotate[i])
            print(round((i / len(self.rotate) * 100), 1), "%")
        self.A=A
        print("finise calculate kenematics ")



class kin_fun:
    
    def __init__(self, filename):

        print("loding data from {} ... ".format(filename+'.yaml'))

        self.filename=filename

        with open(self.filename+'.yaml', 'r') as ymlfile:
            data = yaml.load(ymlfile)
            
        self.length = np.array(data['arminf']['Length'].split(','),dtype=np.float32 )
        self.offset = np.array(data['arminf']['Offset'].split(','),dtype=np.float32)
        self.limit = np.array(data['arminf']['Limit'].split(','),dtype=np.float32)
        self.fas =np.array(data['arminf']['Fas'].split(','),dtype=np.float32)

        self.kenstart1 = np.array([data['armpos']['arm1'].split(',')],dtype=np.float32)
        self.kenstart2 = np.array([data['armpos']['arm2'].split(',')],dtype=np.float32)
        
        self.maxjoint=np.max(self.length)/2*1.5
        self.allteta = np.array([0,0,0, 0, 0, 0])

        
        self.reverearm = np.array([-1,-1,1])

        print("finise loding data")


# =============================================================================
#     def arm_plot(self, all_joint,i,obstacle):
#         fig = plt.figure(i)
#         ax = fig.gca(projection='3d')
#         ax.set_xlim3d(-500, 500)
#         ax.set_ylim3d(-500, 500)
#         ax.set_zlim3d(-50, 500)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.plot(all_joint[:, 0], all_joint[:, 1], all_joint[:, 2])
#         ax.plot(obstacle[:, 0], obstacle[:, 1], obstacle[:, 2],'ro')
#         plt.show()
# =============================================================================

 

    def configure_check(self, newpoint,obstacle):
        
        p = 5
        newpoint = newpoint * np.pi / 180        
        point_joint_1 = self.forword(newpoint[0, :6])*self.reverearm + self.kenstart1
        point_joint_2 = self.forword(newpoint[0, -6:]) + self.kenstart2
        
        nj1= self.get_numeric_joint(point_joint_1, p)
        nj2 = self.get_numeric_joint(point_joint_2, p)
        arm_pos = np.append(nj1[p:], nj2[p:], axis=0)
        

        dis1=1000
        i=len(arm_pos)-1
        

        tmp=np.min(cdist(nj2[-1].reshape(1,3), obstacle, 'euclidean'))
        #self.min_distance(nj2[-1].reshape(1,3),obstacle)
        
        if(tmp<dis1):
           dis1=tmp

        if (dis1  < self.maxjoint / (p-1)*1.2):
            return 0
        
        while(i>0):
            tmp = np.min(cdist(arm_pos[i].reshape(1,3), obstacle, 'euclidean'))
            if (tmp < self.maxjoint / (p-1)*1.2):
                return 0      
            if(tmp<dis1):
                dis1=tmp
            i=i-min(8,max(int(tmp / 70), 1))
            


        if (dis1  < self.maxjoint / (p-1)*1.2):
            return 0
        
        dis2 = np.min(cdist(nj1, nj2, 'euclidean'))     
        if (dis2  < self.maxjoint / (p-1)*1.5):
            return 0

        return min(dis1,dis2)

    def configure_check_gpu(self, newpoint,obstacle):
        p = 5
        newpoint = newpoint * np.pi / 180
        
       
        point_joint_1 = self.forword(newpoint[0, :6])*self.reverearm + self.kenstart1
        point_joint_2 = self.forword(newpoint[0, -6:]) + self.kenstart2
        
        nj1= self.get_numeric_joint(point_joint_1, p)
        nj2 = self.get_numeric_joint(point_joint_2, p) 
        
        cuda = th.device('cuda')     # Default CUDA device
        arms_th=th.from_numpy(np.append(nj1[p:], nj2[p:], axis=0)).to(device=cuda)
        
        i=len(arms_th)-1
        dis_th=10000
        
# =============================================================================
#                            
#         differences = th.pow((obstacle.unsqueeze(0) - arms_th[-1].reshape(1,3)).to(device=cuda),2).to(device=cuda)
#         dis_th=(th.min(th.sqrt(th.sum(differences , -1).to(device=cuda)).to(device=cuda)).to(device=cuda))            
# 
#         differences = th.pow((obstacle.unsqueeze(0) - arms_th[-1].reshape(1,3)).to(device=cuda),2).to(device=cuda)
#         tmp=(th.min(th.sqrt(th.sum(differences , -1).to(device=cuda)).to(device=cuda)).to(device=cuda))            
#         
#         if(dis_th>tmp):
#             dis_th=tmp
#         if (dis_th < self.maxjoint / (p-1)*1.2):
#                 return 0
# =============================================================================


        while(i>0):
            differences = th.pow((obstacle - arms_th[i]), 2)
            tmp=(th.min(th.sqrt(th.sum(differences , -1))))
            if(tmp<dis_th):
                dis_th=tmp
            if (dis_th < self.maxjoint / (p-1)*1.2):
                return 0
            i=i- min(7,max(int(tmp / 70), 1))
        
        dis2 = np.min(cdist(nj1, nj2))

        if (dis_th < self.maxjoint / (p-1)*1.2):
            return 0
        if ( dis2< self.maxjoint / (p-1)*1.5):
            return 0

        return min(int(dis_th),dis2)

    def forword(self,t):
        l=self.length
# =============================================================================
#         #l=[159,234,42.4264,228,143,0]
#         #
#         A01 = self.Az(t[0])
#         A02 = A01 *  (self.Ad([0, 0, l[0]]) * self.Ay(t[1]))
#         A022 = A02 * (self.Ad([0, 0, l[1]])* self.Ay(pi / 4))
#         A03 = A022 * (self.Ad([0, 0, l[2]]) * self.Ay(t[2]))
#         A033 = A03 * (self.Ad([0, 0, l[2]]) * self.Ay(pi / 4))
#         A04 = A033 *  (self.Ad([0, 0, l[3]])* self.Az(t[3]))
#         A05 = A04 *  (self.Ay(t[4]) * self.Ad([0, 0, l[4]]))
#         A06 =  (A05 *self.Az(t[5]))
#         
# # =============================================================================
# #         A01 =sp.trigsimp( self.Az(self.sym_t[0]))
# #         A02 = sp.trigsimp(A01 * self.Ad([0, 0, self.sym_l[0]]) * self.Ay(self.sym_t[1]))
# #         A022 =sp.trigsimp( A02 * self.Ad([0, 0, self.sym_l[1]]) * self.Ay(pi / 4))
# #         A03 = sp.trigsimp(A022 * self.Ad([0, 0, self.sym_l[2]]) * self.Ay(self.sym_t[2]))
# #         A033 =sp.trigsimp( A03 * self.Ad([0, 0, self.sym_l[2]]) * self.Ay(pi / 4))
# #         A04 = sp.trigsimp(A033 * self.Ad([0, 0, self.sym_l[3]]) * self.Az(self.sym_t[3]))
# #         A05 = sp.trigsimp(A04 * self.Ay(self.sym_t[4]) * self.Ad([0, 0, self.sym_l[4]]))
# #         A06 = sp.trigsimp(A05 * self.Az(self.sym_t[5]))
# # =============================================================================
#         self.zerovec=np.array([[0],[0],[0],[1]])
#         points=np.dot(A06 , self.zerovec).T
#         points=np.append(points , np.dot(A04 , self.zerovec).T,axis=0)
#         points=np.append(points , np.dot(A033 , self.zerovec).T,axis=0)
#         points=np.append(points , np.dot(A03 , self.zerovec).T,axis=0)
#         points=np.append(points , np.dot(A022 , self.zerovec).T,axis=0)
#         points=np.append(points , np.dot(A02 ,self.zerovec).T,axis=0)
#         points= np.append(points ,np.dot(A01,self.zerovec).T,axis=0)
#         points=points[:,:3]
#         R06=np.append(np.array(A06[:3,:3]),np.array([[0],[0],[0]]),axis=1)
#         R06=np.append(R06,np.array([[0,0,0,1]]),axis=0)
#         #
#         # R033=A033[:3,:3]
# 
# =============================================================================

        points=np.array([[l[4]*(-(sin(t[0])*sin(t[3]) + sin(t[1] + t[2])*cos(t[0])*cos(t[3]))*sin(t[4]) + cos(t[0])*cos(t[4])*cos(t[1] + t[2])) + (l[1]*sin(t[1]) + l[2]*sin(t[1] + pi/4) + l[2]*sin(t[1] + t[2] + pi/4) + l[3]*cos(t[1] + t[2]))*cos(t[0]),
                            l[4] * ((-sin(t[0]) * sin(t[1] + t[2]) * cos(t[3]) + sin(t[3]) * cos(t[0])) * sin(t[4]) + sin(t[0]) * cos(t[4]) * cos(t[1] + t[2])) + (l[1] * sin(t[1]) +  l[2] * sin(t[1] + pi / 4) +  l[2] * sin(t[1] + t[2] + pi / 4) + l[3] * cos(t[1] + t[2])) * sin(t[0]),
                            l[0] + l[1] * cos(t[1]) + l[2] * (-0.7071 * sin(t[1]) + 0.7071 * cos(t[1])) +  l[2] * cos(t[1] + t[2] + pi / 4) -  l[3] * sin(t[1] + t[2]) -  l[4] * (sin(t[4]) * cos(t[3]) * cos(t[1] + t[2]) + sin(t[1] + t[2]) * cos(t[4]))],

                          [(l[1]*sin(t[1]) +l[2]*sin(t[1] + pi/4) + l[2]*sin(t[1] + t[2] + pi/4) + l[3]*cos(t[1] + t[2]))*cos(t[0]),
                           (l[1] * sin(t[1]) +  l[2] * sin(t[1] + pi / 4) +  l[2] * sin(t[1] + t[2] + pi / 4) + l[3] * cos(t[1] + t[2])) * sin(t[0]),
                           l[0] + l[1] * cos(t[1]) + l[2] * (-0.7071 * sin(t[1]) + 0.7071 * cos(t[1])) +  l[2] * cos(t[1] + t[2] + pi / 4) - l[3] * sin(t[1] + t[2])],

                          [(1.0*l[1]*sin(t[1]) + l[2]*sin(t[1] + pi/4) + l[2]*sin(t[1] + t[2] + pi/4))*cos(t[0]),
                           (1.0 * l[1] * sin(t[1]) +  l[2] * sin(t[1] + pi / 4) +  l[2] * sin(t[1] + t[2] + pi / 4)) * sin(t[0]),
                           l[0] + l[1] * cos(t[1]) + l[2] * (-0.7071 * sin(t[1]) + 0.7071 * cos(t[1])) +  l[2] * cos(t[1] + t[2] + pi / 4)],

                          [(1.0 * l[1] * sin(t[1]) +  l[2] * sin(t[1] + pi / 4)) * cos(t[0]),
                          (1.0 * l[1] * sin(t[1]) +  l[2] * sin(t[1] + pi / 4)) * sin(t[0]),
                          l[0] + l[1] * cos(t[1]) + l[2] * (-0.7071 * sin(t[1]) + 0.7071 * cos(t[1]))],

                          [l[1]*sin(t[1])*cos(t[0]),
                          l[1]*sin(t[0])*sin(t[1]),
                          l[0] + l[1]*cos(t[1])],

                          [0,0,l[0]],

                          [0,0,0]])

        return points





    def get_numeric_joint(self, tempvec,num):
        numpoints=( np.zeros((1,3)))
        for i in range(1, len(tempvec)):
            numpoints=np.append(np.linspace(tempvec[i],tempvec[i-1],num),numpoints[1:],axis=0)
        return numpoints
    
 

    def invers_arms_cfg(self,point):
        p1 = point[0:3] + self.kenstart2
        p2 = point[3:6] + self.kenstart1
        o1 = point[6:9]
        o2 = point[9:12]
        p1[0, 1] = p1[0, 1] * -1
        p1[0, 0] = p1[0, 0] * -1
        a1 = self.invers(p1, o1, 0)
        a2 = self.invers(p2, o2, 0)
        return np.append(a1, a2, axis=1)

    def invers(self,pos,ori,i=0,t6=0,t1=0):
        l=[159,265.6991,259.7383,123]
        fas=[1.4576,1.3419]
        s=np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]])

        #R06=self.Ax(ori[0])*self.Ay(ori[1])*self.Az(ori[2])*self.Ay(pi/2)
        R06=np.array([[-sin(ori[1]),-sin(ori[2])*cos(ori[1]),cos(ori[1])*cos(ori[2]),0],
                      [sin(ori[0])*cos(ori[1]),-sin(ori[0])*sin(ori[1])*sin(ori[2]) + cos(ori[0])*cos(ori[2]),sin(ori[0])*sin(ori[1])*cos(ori[2])+ sin(ori[2])*cos(ori[0]),0],
                      [-cos(ori[0])*cos(ori[1]),sin(ori[0])*cos(ori[2]) + sin(ori[1])*sin(ori[2])*cos(ori[0]),sin(ori[0])*sin(ori[2]) -sin(ori[1])*cos(ori[0])*cos(ori[2]),0],[0,0,0,1]])

        #pc=self.Ad([pos[0],pos[1],pos[2]])*np.dot(R06,np.array([[0],[0],[-l[3]],[1]])))
        pc=np.array([[pos[0,0] -l[3]*cos(ori[1])*cos(ori[2])],
                    [pos[0,1] - l[3]*(sin(ori[0])*sin(ori[1])*cos(ori[2]) + sin(ori[2])*cos(ori[0]))],
                    [pos[0,2] - l[3]*(sin(ori[0])*sin(ori[2]) - sin(ori[1])*cos(ori[0])*cos(ori[2]))]])

        R06=R06[:3,:3]
        gova=pc[2,0]-l[0]
        yeter=s[i,0]*(pc[0,0]**2+pc[1,0]**2)**0.5
        D=((yeter**2+gova**2-l[1]**2-l[2]**2)/(2*l[1]*l[2]))

        if(pc[0,0]==0 and pc[1,0]==0):
            t1=t1
        else:
            t1=np.arctan2(pc[1,0]/(yeter),pc[0,0]/(yeter))
        t3=np.arctan2(s[i,1]*(1-D**2)**0.5,D)
        t2=-(np.arctan2(gova,yeter) - np.arctan2(l[2]*sin(-t3),l[1]+l[2]*cos(-t3)))
        t3 =t3 - fas[1]
        t2 =t2 + fas[0]


        R03=np.array([[-sin(t2 + t3)*cos(t1) , -sin(t1) , cos(t1)*cos(t2 + t3) ],
                       [-sin(t1)*sin(t2 + t3) , cos(t1)  , sin(t1)*cos(t2 + t3)],
                       [-cos(t2 + t3)         , 0       ,  -sin(t2 + t3)]])


        # http://aranne5.bgu.ac.il/Free_publications/mavo-lerobotica.pdf
        R36=np.dot(R03.T,R06)

        st5=-s[i,2]*(R36[0,2]**2+R36[1,2]**2)**0.5
        if abs(st5)<1e-4:
            t5=0
            t6 = t6
            t4=np.arctan2(R36[1,0],R36[0,0])+t6
        else:
            t5=np.arctan2(st5,R36[2,2])
            t4=np.arctan2(R36[1,2]/st5,R36[0,2]/st5)
            t6=np.arctan2(R36[2,1]/st5,-R36[2,0]/st5)

        return (np.array([[t1,t2,t3,t4,t5,t6]])*180/pi)


# a = rotation_matrices('/home/koby/ws_moveit/src/rrt/include/ManipulatorH')
# #p = np.array(a.forword([0.4,-0.2,0.6,0.1,0.4,-0.7]))
# ori=np.array([0,0,0])
# c=np.array([[0,0,0,0,0,0]])
# ind=0
# for i in range(100):
#     p=np.array([[260-i*2,0,410]])
#     if(not i==0):
#         oldc = c
#         print(c)
#         for j in range(8):
#             c= a.invers(p[0], ori, ind, c[0,5], c[0,1])
#             if max(abs(abs(oldc[0])-abs(c[0])))>3:
#                 ind=j
#                 c=oldc
#             else:
#                 break
#     else:
#         c = a.invers(p[0], ori, ind, c[0, 5], c[0, 1])
#     b=ros(100)
#     time.sleep(0.01)
#
#     t=np.append(c, c, axis=1)
#     b.send_to_arm(t)
