import numpy as np
from script.utils import Node, Utils, FloydWarshall
from script.world import WorldMap
import pickle
import matplotlib.pyplot as plt

from time import time, sleep
from random import randint


class Mwrp:
    def __init__(self, world, start_pos):
        self.start_time = 0
        self.number_of_agent = start_pos.__len__()

        self.world = world

        self.node_expend_index = 1

        all_free_cells=set(map(tuple, np.transpose(np.where(self.world.grid_map == 0))))

        need_to_see_at_start = all_free_cells - self.world.get_all_seen( start_pos)

        start_node = Node(Node(None, start_pos, 0), start_pos, need_to_see_at_start)

        self.open_list = [start_node]

        self.visit_list_dic = {tuple(sorted(start_pos)): [start_node]}

        self.number_of_node = 0
        tmp = time()
        print("\nstart calculate distance ... ", end='')

        self.real_dis_dic = pickle.load(open('2.p', 'rb'))
        dist= pickle.load(open('1.p', 'rb'))

        self.real_dis_dic, dist = FloydWarshall(self.world.grid_map).floyd_warshall()
        #pickle.dump(dist, open('1.p', 'wb'))
        #pickle.dump(self.real_dis_dic, open('2.p', 'wb'))

        print(f"finise at {time() - tmp} sec ")

        # centrality(dist,self.world.grid_map,self.world.dict_wachers)
        # centrality_list=Utils.centrality_meen_see(self.world.dict_wachers)
        # centrality_list=Utils.centrality_tie_see(dist,self.world.grid_map,self.world.dict_wachers)
        # centrality_list=Utils.centrality_tie_wahcers(dist,self.world.grid_map,self.world.dict_wachers)
        # centrality_list=Utils.centrality(dist,self.world.grid_map)

        print("\nstart calculate centrality ... ", end='')
        tmp = time()
        self.centrality_dict = self.centrality_list_wachers(dist,all_free_cells,dist)
        print(f"finise at {time() - tmp} sec ")

        print("\nstart calculate pivot ... ", end='')
        tmp = time()
        self.pivot = self.get_pivot(set(map(tuple,all_free_cells)),self.centrality_dict)
        self.print_pivot()
        print(f"finise at {time() - tmp} sec ")



    def centrality_list_wachers(self,dist_dict,free_cells,dist):

        centrality_dict = []
        for index, cell in enumerate(free_cells):
            centrality_value=0
            for cell_whacers in self.world.dict_wachers[cell]:
                centrality_value += dist_dict[cell_whacers]
            centrality_dict.append((centrality_value/(self.world.dict_wachers[cell].__len__()**10)*dist[cell],cell))
            centrality_dict=sorted(centrality_dict,reverse=True)
            #dist_dict[cell]=
        return centrality_dict


    def get_pivot(self, need_to_see,a):
        pivot = dict()

        # tmp_centrlize_list = []
        # for cell in need_to_see:
        #     tmp_centrlize_list.append((self.centrality_dict[cell], cell))
        # tmp_centrlize_list = sorted(tmp_centrlize_list, reverse=True)
        #self.print_whacers([(1,4),(2,4),(3,4),(4,4)])
        #exit(0)
        tmp_centrlize_list=a
        for cell in tmp_centrlize_list:
            disjoint = True
            for value in pivot:
                #self.print_whacers(pivot,cell)
                if self.world.dict_wachers[cell[1]].intersection(pivot[value]).__len__():
                    disjoint = False
                    break
            if disjoint:
                pivot[cell[1]] = self.world.dict_wachers[cell[1]]

        return pivot

    def get_new_node(self, new_state, seen_state, new_cost, old_state):

        heuristic = self.get_heuristic(new_state, seen_state)
        new_node = Node(old_state, new_state, seen_state, new_cost, heuristic)
        return new_node

    # def find_index_to_open1(self, new_node):
    #     all_cost_estimate = new_node.cost + new_node.heuristics
    #     for index, data in enumerate(self.open_list):
    #         if data.cost + data.heuristics == all_cost_estimate:
    #             if data.cost == new_node.cost:
    #                 if data.need_to_see.__len__() > new_node.need_to_see.__len__():
    #                     return index
    #
    #             # elif data.need_to_see.__len__() == new_node.need_to_see.__len__():
    #             elif data.cost > new_node.cost:
    #                 return index
    #         elif data.cost + data.heuristics > all_cost_estimate:
    #             return index
    #     return len(self.open_list)

    def find_index_to_open(self, new_node):
        all_cost_estimate = new_node.cost + new_node.heuristics
        for index, data in enumerate(self.open_list):
            if data.cost + data.heuristics == all_cost_estimate:
                if data.need_to_see.__len__() > new_node.need_to_see.__len__():
                    return index
                elif data.need_to_see.__len__() == new_node.need_to_see.__len__():
                    if data.cost < new_node.cost:
                        return index
            elif data.cost + data.heuristics > all_cost_estimate:
                return index

        return len(self.open_list)

    def insert_to_open_list(self, new_node):
        new_node.heuristics = self.get_heuristic(new_node)
        index = self.find_index_to_open(new_node)
        self.open_list.insert(index, new_node)
        return new_node

    def pop_open_list(self):

        if len(self.open_list):
            pop_open_list = self.open_list.pop(0)
        else:
            pop_open_list = 0

        return pop_open_list

    def get_real_dis(self, cell_a, cell_b):
        key = tuple(sorted((cell_a, cell_b)))

        if key in self.real_dis_dic:
            return self.real_dis_dic[key]
        print('no key !!!!')
        return -1

    def singelton_heuristic(self, new_node):
        max_pivot_dist = 0
        for cell in new_node.need_to_see:
            min_dis = 1000000
            for whach in self.world.dict_wachers[cell]:
                if min_dis < max_pivot_dist:
                    break
                for agent in new_node.location:
                    real_dis = self.get_real_dis(agent, whach)
                    if real_dis < min_dis:
                        min_dis = real_dis
            if max_pivot_dist < min_dis:
                max_pivot_dist = min_dis

        return max_pivot_dist

    def get_heuristic(self, new_node):
        closest_pivot_dist = self.singelton_heuristic(new_node)
        return closest_pivot_dist

    def print_pivot(self):
        tmp = np.copy(self.world.grid_map)
        for cell in self.pivot.keys():
            tmp[cell] = 3
        plt.figure(1)
        plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
        plt.show()
        x = 1

    def print_whacers(self, tmp_cell):

        for pivot_cell in tmp_cell:
            tmp = np.copy(self.world.grid_map)

            for cell in self.world.dict_wachers[pivot_cell]:
                tmp[cell] = 3
            tmp[pivot_cell] = 2

            plt.figure(pivot_cell.__str__())
            plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
            plt.gca().set_aspect('equal')
            plt.gca().set_ylim(plt.gca().get_ylim()[::-1])


        plt.show()

    def print_whacers1(self,pivot_cell,tmp_cell):
        tmp = np.copy(self.world.grid_map)

        for pivot_cell in pivot_cell.keys():
            for cell in self.world.dict_wachers[pivot_cell]:
                tmp[cell] = 3
            tmp[pivot_cell] = 2

        for cell in self.world.dict_wachers[tmp_cell[1]]:
            if tmp[cell] == 3:
                tmp[cell] = -1
            else:
                tmp[cell] = 3


        tmp[tmp_cell[1]] = -2


        plt.figure(1)
        plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
        plt.show()

    def print_path(self, gole_node):
        all_path = self.get_path(gole_node)

        for cell in all_path:
            # print(f'{cell.location}  H = {cell.heuristics}')
            tmp_word = np.copy(self.world.grid_map)
            for k in cell.need_to_see:
                tmp_word[k] = 2
            for j in cell.location:
                tmp_word[j] = 3

            plt.figure(1)
            plt.pcolormesh(tmp_word, edgecolors='grey', linewidth=0.01)
            plt.gca().set_aspect('equal')
            plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
            plt.draw()
            plt.pause(0.001)
            plt.clf()
            sleep(0.5)

    def get_new_state(self, old_state, state_index):

        move_index = [0] * self.number_of_agent
        for j in range(self.number_of_agent):
            state_index, index = divmod(state_index, LOS)
            move_index[j] = index

        new_state = self.world.get_action(old_state.location, move_index)
        return new_state

    def get_cost(self, old_state):
        return old_state.cost + 1

    def in_open_or_close(self, new_node):  # old_state, new_state, seen_state):
        tmp_state = tuple(sorted(new_node.location))
        if not tmp_state in self.visit_list_dic:
            return False
        return True

    def expend(self):
        old_state = self.pop_open_list()
        for state_index in range(LOS ** self.number_of_agent):

            new_state = self.get_new_state(old_state, state_index)

            if self.world.is_valid_node(new_state, old_state):

                self.number_of_node += 1

                seen_state = old_state.need_to_see - self.world.get_all_seen(new_state)

                if self.goal_test(seen_state):
                    new_node = Node(old_state, new_state, seen_state, old_state.cost + 1, 0)
                    print(f"find solution in -> {time() - self.start_time} sec at cost of {new_node.cost}"
                          f" and open {self.number_of_node} node")
                    return new_node

                new_cost = self.get_cost(old_state)

                new_node = Node(old_state, new_state, seen_state, new_cost, 0)

                if not self.in_open_or_close(new_node):  # old_state, new_state, seen_state):
                    self.insert_to_open_list(new_node)
                    self.visit_list_dic[tuple(sorted(new_node.location))] = [new_node]
                elif self.need_to_fix_parent(new_node):  # new_state, seen_state, old_state):
                    self.insert_to_open_list(new_node)
                    # new_node = self.get_new_node(new_state, seen_state, new_cost, old_state)
                    self.visit_list_dic[tuple(sorted(new_node.location))].append(new_node)

        return False

    def need_to_fix_parent(self, new_node):
        tmp_state = tuple(sorted(new_node.location))

        for old_node in self.visit_list_dic[tmp_state]:
            if new_node.cost >= old_node.cost and old_node.need_to_see.issubset(new_node.need_to_see):
                new_node.parent = old_node.parent
                new_node.cost = old_node.cost
                return False

            elif new_node.cost <= old_node.parent.cost and new_node.need_to_see.issubset(old_node.need_to_see):
                old_node.parent = new_node.parent
                old_node.cost = new_node.cost
                return False

        return True

    def goal_test(self, map_state):
        if not map_state.__len__():
            return True
        return False

    def run(self):
        self.start_time = time()
        print("\nstart algoritem ... ", end='')

        gole_node = False
        while not gole_node:
            gole_node = self.expend()
            continue
        input()
        self.print_path(gole_node)

    def get_path(self, gole_node):
        all_path = [gole_node]
        node = gole_node
        while node.parent.parent is not None:
            node = node.parent
            all_path.append(node)
        return all_path[::-1]


if __name__ == '__main__':
    map_type = 'big_'
    map_config = './config/map_a_{}config.csv'.format(map_type)
    row_map = np.genfromtxt(map_config, delimiter=',', case_sensitive=True)

    LOS = 4
    start_pos = ((1, 1), (1, 15))  # ,(2,10))
    world = WorldMap(row_map, LOS)
    mwrp = Mwrp(world, start_pos)

    # pickle.dump(mwrp.real_dis_dic, open(map_type + '.p', 'wb'))
    mwrp.run()


# TODO:
# heuristic for all robot by A* graf

# find pivot by cenralty sum on all dis -> V
# fix state as world + locashon -V
# improve g swich to seen chack -> V
# movment index  insted of a list -> V
# bild seen fun for all cell obove the tree -V
# doplicate ((1,0)(0,1))==((0,1)(1,0)) -> V
# bild real distance travet dict ->V
# heuristic sigmoind ->v
# whacers / los Bresenham -> V
# make state and seen by set() -> V
# fix f prblom line f1=f2 - > V
