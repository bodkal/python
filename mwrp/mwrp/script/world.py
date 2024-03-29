import numpy as np
from itertools import product
from time import time
import sys
from script.utils import Bresenhams ,Utils

class WorldMap:

    def __init__(self, row_map,los):
        self.grid_map = row_map.astype(np.int8)
        self.breas = Bresenhams(self.grid_map)


        [self.col_max,self.row_max] = self.grid_map.shape
        self.free_cell=len((np.where(self.grid_map==0)[0]))

        #self.LOS=los
        self.dict_wachers=dict()
        self.dict_fov=dict()

        self.create_wachers()


        if(los==9):
            self.action = self.get_static_action_9(1)
        elif(los==8):
            self.action = self.get_static_action_8(1)
        elif(los==5):
            self.action = self.get_static_action_5(1)
        elif(los==4):
            self.action = np.vstack((self.get_static_action_4(1),[0,0]))


    def get_action(self, state, move_index):

        action=[]
        moving_status=[]
        for index,data in enumerate(state):
                action.append((self.action[move_index[index]][0]+ data[0],self.action[move_index[index]][1]+ data[1]))
                moving_status.append(np.sum(np.abs(self.action[move_index[index]])))
        return tuple(action) , moving_status

    def is_disjoit(self,cell_a,cell_b):
        if self.dict_wachers[cell_a].intersection(self.dict_wachers[cell_b]):
            return False
        return True

    def in_bund(self, state):
        for cell in state:
            if cell[0] > self.col_max-1 or cell[1] > self.row_max-1 or cell[0]<0 or cell[1]<0:
                return False
        return True

    def is_obstical(self,state):
        for cell in state:
            if self.grid_map[cell]==1:
                return False
        return True

    def get_static_action_9(self, agents_number):
        return np.reshape(list(product([1, 0, -1], repeat=agents_number * 2)), (-1, agents_number * 2)).astype(int)

    def get_static_action_8(self, agents_number):
        all_opsens = np.reshape(list(product([1, 0, -1], repeat=agents_number * 2)), (-1, agents_number * 2))
        all_index = np.arange(9 ** agents_number)
        for i in range(agents_number):
            tmp_index = np.where(np.any(all_opsens[:, i * 2: i * 2 + 2], axis=1))
            all_index = np.intersect1d(tmp_index, all_index)
        return np.copy(all_opsens[all_index]).astype(int)

    def get_static_action_5(self, agents_number):
        all_opsens = np.reshape(list(product([1, 0, -1], repeat=agents_number * 2)), (-1, agents_number * 2))
        all_index = np.arange(9 ** agents_number)
        for i in range(agents_number):
            tmp_index = np.where(np.any(all_opsens[:, i * 2: i * 2 + 2] == 0, axis=1))
            all_index = np.intersect1d(tmp_index, all_index)
        return np.copy(all_opsens[all_index]).astype(int)

    def get_static_action_4(self, agents_number):
        all_opsens = np.reshape(list(product([1, 0, -1], repeat=agents_number * 2)), (-1, agents_number * 2))
        all_index = np.arange(9 ** agents_number)
        for i in range(agents_number):
            tmp_index = np.where(np.abs(all_opsens[:, i * 2]) != np.abs(all_opsens[:, i * 2 + 1]))
            all_index = np.intersect1d(tmp_index, all_index)
        return np.copy(all_opsens[all_index]).astype(int)

    # def los_4(self, state):
    #
    #     los4=np.zeros((1,2)).astype(int)
    #     for i in range(state[1]+1,self.row_max):
    #         if self.grid_map[state[0],i] == 1:
    #             break
    #         los4=np.vstack((los4,[state[0],i]))
    #
    #     for i in np.flip(range(state[1])):
    #         if self.grid_map[state[0],i] == 1:
    #             break
    #         los4=np.vstack((los4,[state[0],i]))
    #
    #     for i in range(state[0],self.col_max):
    #         if self.grid_map[i,state[1]] == 1:
    #             break
    #         los4=np.vstack((los4,[i,state[1]]))
    #
    #     for i in np.flip(range(state[0])):
    #         if self.grid_map[i,state[1]] == 1:
    #             break
    #         los4=np.vstack((los4,[i,state[1]]))
    #
    #     return los4[1:]
    #
    # def los_4_test(self, state,tmp_map):
    #
    #     for i in range(state[1] + 1, self.row_max):
    #         if self.grid_map[state[0], i] == 1:
    #             break
    #         tmp_map[state[0], i]=2
    #
    #     for i in np.flip(range(state[1])):
    #         if self.grid_map[state[0], i] == 1:
    #             break
    #         tmp_map[state[0], i] = 2
    #
    #     for i in range(state[0], self.col_max):
    #         if self.grid_map[i, state[1]] == 1:
    #             break
    #         tmp_map[i, state[1]]=2
    #
    #     for i in np.flip(range(state[0])):
    #         if self.grid_map[i, state[1]] == 1:
    #             break
    #         tmp_map[i, state[1]]=2
    #
    #     return tmp_map
    #
    #
    # def los_8(self, state):
    #     los8=self.los_4(state)
    #     for i in range(1,self.row_max):
    #         if self.grid_map[state[0]+i,state[1]+ i] == 1:
    #             break
    #         los8=np.vstack((los8,[state[0]+i,state[1]+ i]))
    #
    #     for i in range(1,min(state[0],state[1])):
    #         if self.grid_map[state[0]-i,state[1]- i] == 1:
    #             break
    #         los8=np.vstack((los8,[state[0]-i,state[1]- i]))
    #
    #     for i in range(1,self.col_max):
    #         if self.grid_map[state[0]-i,state[1]+ i] == 1:
    #             break
    #         los8=np.vstack((los8,[state[0]-i,state[1]+ i]))
    #
    #     for i in range(1,max(state[0],state[1])):
    #         if self.grid_map[state[0]+i,state[1] - i] == 1:
    #             break
    #         los8=np.vstack((los8,[state[0]+i,state[1]- i]))
    #
    #     return los8
    #
    #
    # def get_seen_from_cell(self,state):
    #     seen=0
    #     if (self.LOS == 9):
    #         seen=self.los_9(state)
    #     elif (self.LOS  == 8):
    #         seen=self.los_8(state)
    #     elif (self.LOS  == 5):
    #         seen=self.los_5(state)
    #     elif (self.LOS  == 4):
    #         seen=self.los_4(state)
    #     return set(map(tuple,seen))

    # def get_seen_from_cell_breas(self,state):
    #     return self.get_fov()




    # def create_wachers(self):
    #     all_free_cell=set(map(tuple,np.asarray(np.where(self.grid_map == 0)).T))
    #     for cell in all_free_cell:
    #         self.dict_wachers[cell] = self.get_fov(cell)




    def create_wachers(self):
        all_free_cell = set(map(tuple, np.asarray(np.where(self.grid_map == 0)).T))
        for cell in all_free_cell:
            tmp_set = self.get_fov(cell)
            #if cell == (7,13):
            #Utils.print_fov(self.grid_map,tmp_set,cell)
            self.dict_fov[cell]=tmp_set
            for wahers in tmp_set:
                if wahers != cell:
                    if wahers in self.dict_wachers:
                        self.dict_wachers[wahers] = self.dict_wachers[wahers].union(Utils.map_to_sets(cell))
                    else:
                        self.dict_wachers[wahers] = Utils.map_to_sets(cell)
        # Utils.print_fov(self.grid_map,self.get_fov((6,6)), (6,6))



    def get_all_seen(self,state):
        tmp_set_seen=set()
        for cell in state:
            tmp_set_seen = tmp_set_seen.union(self.dict_fov[cell])
            #tmp_set_seen = tmp_set_seen.union(self.get_fov(cell))

        return tmp_set_seen


    def is_valid_node(self, new_state, old_state,moving_status):

        if not self.in_bund(new_state):
            return False
        elif not self.is_obstical(new_state):
            return False
        elif new_state == old_state.parent.location:
            return False
        elif not np.any(moving_status):
            return False

        for i in old_state.dead_agent:
            if moving_status[i]!=0:
                return False
        return True



    def get_fov(self,start_cell):
        all_seen = set()

        for y in range(self.grid_map.shape[1]):
            end_cell=(0,y)
            tmp_seen = self.breas.get_line(start_cell, end_cell)
            all_seen = all_seen | tmp_seen

        for x in range(1,self.grid_map.shape[0]):
            end_cell = (x,0)
            tmp_seen = self.breas.get_line(start_cell, end_cell)
            all_seen = all_seen | tmp_seen

        for x in range(1,self.grid_map.shape[0]):
            end_cell = (x,self.grid_map.shape[1]-1)
            tmp_seen = self.breas.get_line(start_cell, end_cell)
            all_seen = all_seen | tmp_seen

        for y in range(self.grid_map.shape[1]):
            end_cell=(self.grid_map.shape[0]-1,y)
            tmp_seen= self.breas.get_line(start_cell, end_cell)
            all_seen= all_seen | tmp_seen



        return all_seen




