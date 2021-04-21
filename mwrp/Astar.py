import numpy as np
from script.utils import Node, Utils
from script.world import WorldMap
from time import time

class Astar:
    def __init__(self, world, start, need_to_be):


        self.world = world
        start=Node(Node(None, start, 0), start, need_to_be)
        self.visit_list_dic = {tuple(sorted(start)): [start]}
        self.open_list = [start]



    def insert_to_open_list(self, new_node, old_node):
        cost_estimate = new_node.cost + new_node.heuristics
        for index, data in enumerate(self.open_list):
            if data.cost + data.heuristics > cost_estimate:
                self.open_list = np.insert(self.open_list, index, new_node)
                self.open_list_dic[new_node.Location.__str__()] = old_node
                return

        if (len(self.open_list)):
            self.open_list = np.hstack((self.open_list, new_node))
        else:
            self.open_list = np.array([new_node])
        self.open_list_dic[new_node.Location.__str__()] = old_node

    def pop_open_list(self):
        pop_open_list = self.open_list[0]
        self.open_list = self.open_list[1:]
        self.close_list_dic[pop_open_list.Location.__str__()] = self.open_list_dic[pop_open_list.Location.__str__()]
        del self.open_list_dic[pop_open_list.Location.__str__()]
        return pop_open_list

    def move_from_open_to_close(self, index=0):
        self.open_list = np.delete(self.open_list, index)

    def get_heuristic(self, state):
        return sum(abs(self.goal-state)) #np.linalg.norm(state - self.goal)

    def get_action(self, state, move_index):
        action = []
        for i in move_index:
            action = np.hstack((action, self.action[i]))

        return (action).astype(int) + state

    def get_path(self, gole_node):
        all_path = np.array([self.goal])
        node = gole_node
        while node.cost:
            print(node.Location,end='')
            all_path = np.vstack((all_path, node.Location))
            node = self.close_list_dic[node.Location.__str__()]
        all_path = np.vstack((all_path, self.start))

        return all_path

    def run(self, start=0, goal=0):

        def run(self):
            self.start_time = time()
            print("\nstart algoritem ... ", end='')

            gole_node = False
            while not gole_node:
                gole_node = self.expend()
                continue
        _, cost=self.get_path()

        return cost

    def get_path(self, gole_node):
        all_path = [gole_node]
        node = gole_node
        cost=0
        while node.parent.parent is not None:
            cost+=node.cost
            node = node.parent
            all_path.append(node)
        return all_path[::-1] ,cost

    def expend(self):
        old_state = self.pop_open_list()
        for state_index in range(self.LOS ** self.number_of_agent):

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

    def goal_test(self, map_state):
        if not map_state.__len__():
            return True
        return False



class Node:

    def __init__(self, Location, cost=0, heuristics=0):
        self.Location = Location
        self.cost = cost
        self.heuristics = heuristics
