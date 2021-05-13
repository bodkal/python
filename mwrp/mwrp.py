import numpy as np
from script.utils import Node, Utils, FloydWarshall
from script.world import WorldMap
import pickle
import matplotlib.pyplot as plt

from time import time, sleep
from random import randint


class Mwrp:

    def __init__(self, world, start_pos):
        self.a = 0
        self.b = 0
        self.start_time = 0
        self.number_of_agent = start_pos.__len__()

        self.world = world

        self.node_expend_index = 1

        unseen_all = set(map(tuple, np.transpose(np.where(self.world.grid_map == 0))))

        unseen_start = unseen_all - self.world.get_all_seen(start_pos)

        start_node = Node(Node(None, start_pos, 0), start_pos, unseen_start)

        self.open_list = [start_node]

        self.visit_list_dic = {tuple(sorted(start_pos)): [start_node]}

        self.number_of_node = 0

        self.real_dis_dic, self.centrality_dict = FloydWarshall(self.world.grid_map).floyd_warshall()

        print("\nstart calculate pivot ... ", end='')
        tmp = time()
        self.pivot = self.get_pivot(unseen_all)
        # Utils.print_pivot(self.world, self.pivot)
        print(f"finise at {time() - tmp} sec ")

    def goal_test(self, map_state):
        if not map_state.__len__():
            return True
        return False

    def centrality_list_wachers(self, unseen):

        list_sort_by_centrality = []
        for index, cell in enumerate(unseen):
            centrality_value = 0
            for cell_whacers in self.world.dict_wachers[cell]:
                centrality_value += self.centrality_dict[cell_whacers]
            list_sort_by_centrality.append(
                (centrality_value / (self.world.dict_wachers[cell].__len__() ** 2) * self.centrality_dict[cell], cell))
        list_sort_by_centrality = sorted(list_sort_by_centrality, reverse=True)
        return list_sort_by_centrality

    def get_pivot(self, unseen):
        centrality_dict = self.centrality_list_wachers(unseen)
        pivot = dict()
        while unseen.__len__():
            cell = centrality_dict.pop(0)
            if cell[1] in unseen:
                pivot[cell[1]] = self.world.dict_wachers[cell[1]]
                for pivot_wacher in pivot[cell[1]]:
                    unseen = unseen - self.world.dict_wachers[pivot_wacher]
        return pivot

    def get_pivot1(self, unseen):

        centrality_dict = self.centrality_list_wachers(unseen)
        pivot = dict()

        while centrality_dict.__len__():

            cell = centrality_dict.pop(0)
            disjoint = True
            for value in pivot:
                if self.world.dict_wachers[cell[1]].intersection(pivot[value]).__len__():
                    disjoint = False
                    break
            if disjoint:
                pivot[cell[1]] = self.world.dict_wachers[cell[1]]
                for pivot_wacher in pivot[cell[1]]:
                    for wacher_of_wacher in self.world.dict_wachers[pivot_wacher]:
                        if (wacher_of_wacher in centrality_dict):
                            centrality_dict.remove(wacher_of_wacher)

        return pivot

    def get_new_node(self, new_state, seen_state, new_cost, old_state):

        heuristic = self.get_heuristic(new_state, seen_state)
        new_node = Node(old_state, new_state, seen_state, new_cost, heuristic)
        return new_node

    def find_index_to_open(self, new_node):

        index_a = 0
        index_b = len(self.open_list)
        all_cost_estimate = new_node.cost + new_node.heuristics
        while index_a < index_b:
            mid = (index_a + index_b) // 2
            data = self.open_list[mid]
            if all_cost_estimate == data.cost + data.heuristics:

                if data.unseen.__len__() == new_node.unseen.__len__():
                    if data.cost < all_cost_estimate:
                        index_b = mid
                    else:
                        index_a = mid + 1
                elif data.unseen.__len__() > new_node.unseen.__len__():
                    index_b = mid
                else:
                    index_a = mid + 1
            elif all_cost_estimate < data.cost + data.heuristics:
                index_b = mid
            else:
                index_a = mid + 1
        return index_a

    def insert_to_open_list(self, new_node):

        if not self.in_open_or_close(new_node):
            new_node.heuristics = self.get_heuristic(new_node)
            index = self.find_index_to_open(new_node)
            self.open_list.insert(index, new_node)
            self.visit_list_dic[tuple(sorted(new_node.location))] = [new_node]
        elif self.need_to_fix_parent(new_node):
            new_node.heuristics = self.get_heuristic(new_node)
            index = self.find_index_to_open(new_node)
            self.open_list.insert(index, new_node)
            self.visit_list_dic[tuple(sorted(new_node.location))].append(new_node)

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
        # Utils.print_whacers(self.world,new_node.unseen)
        for cell in new_node.unseen:
            min_dis = 1000000
            for whach in self.world.dict_wachers[cell]:

                if min_dis < max_pivot_dist:
                    break

                for agent in new_node.location:
                    real_dis = self.get_real_dis(agent, whach)
                    min_dis = min(min_dis, real_dis)
            max_pivot_dist = max(max_pivot_dist, min_dis)

        return max_pivot_dist

    def get_heuristic(self, new_node):
        closest_pivot_dist = self.singelton_heuristic(new_node)
        return closest_pivot_dist

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
        # print(self.number_of_node,'\t',old_state.heuristics)

        if old_state.cost == -1:
            return False

        if self.goal_test(old_state.unseen):
            return old_state

        for state_index in range(LOS ** self.number_of_agent):

            new_state = self.get_new_state(old_state, state_index)
            if self.world.is_valid_node(new_state, old_state):
                self.number_of_node += 1
                seen_state = old_state.unseen - self.world.get_all_seen(new_state)

                new_node = Node(old_state, new_state, seen_state, self.get_cost(old_state), 0)

                self.insert_to_open_list(new_node)

        return False

    def need_to_fix_parent(self, new_node):
        tmp_state = tuple(sorted(new_node.location))

        for index, old_node in enumerate(self.visit_list_dic[tmp_state]):

            if new_node.cost >= old_node.cost and old_node.unseen.issubset(new_node.unseen):
                return False

            elif new_node.cost <= old_node.cost and new_node.unseen.issubset(old_node.unseen):
                old_node.cost = -1
                del self.visit_list_dic[tmp_state][index]
                return True

        return True

    def run(self):
        self.start_time = time()
        print("\nstart algoritem ... ", end='')

        gole_node = False
        while not gole_node:
            gole_node = self.expend()
            #continue
        print(f"find solution in -> {time() - self.start_time} sec at cost of {gole_node.cost}"
              f" and open {self.number_of_node} node")
        input()
        self.print_path(gole_node, True)

    def get_path(self, gole_node):
        all_path = [gole_node]
        node = gole_node
        while node.parent.parent is not None:
            node = node.parent
            all_path.append(node)
        return all_path[::-1]

    def print_path(self, gole_node, see_agent_walk):
        all_path = self.get_path(gole_node)
        tmp_location = []
        for cell in all_path:
            print(f'{cell.location} ')
            tmp_location.append(cell.location)
            tmp_word = np.copy(self.world.grid_map)
            for k in cell.unseen:
                tmp_word[k] = 2
            for j in cell.location:
                tmp_word[j] = 3
            if see_agent_walk:
                plt.figure(1)
                plt.pcolormesh(tmp_word, edgecolors='grey', linewidth=0.01)
                plt.gca().set_aspect('equal')
                plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
                plt.draw()
                plt.pause(0.001)
                plt.clf()
                sleep(0.5)
        plt.close('all')
        Utils.print_all_whacers(self.world, tmp_location)


if __name__ == '__main__':
    map_type = ''
    map_config = './config/map_a_{}config.csv'.format(map_type)
    row_map = np.genfromtxt(map_config, delimiter=',', case_sensitive=True)

    LOS = 4
    start_pos = ((1, 1), (5, 13),(20,3))
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
