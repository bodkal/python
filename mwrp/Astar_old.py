import numpy as np
# from script.utils import Node, Utils
from script.world import WorldMap


class Astar:
    def __init__(self, world, start, goal):

        self.world = world

        self.start = start
        start = Node(start, 0, 0)
        self.goal = goal

        self.open_list = np.array([start])

        self.open_list_dic = {start.Location.__str__(): start}
        self.close_list_dic = dict()

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

        self.close_list_dic = dict()

        if (np.all(start) or np.all(goal)):
            self.start = start
            self.goal = goal
            start = Node(start, 0, 0)
            self.open_list = np.array([start])
            self.open_list_dic = {start.Location.__str__(): start}

        gole_node = False
        while not gole_node:
            gole_node, cost = self.expend()
            continue
        return self.get_path(gole_node), cost

    def expend(self):
        old_state = self.pop_open_list()

        for new_state in (self.world.action + old_state.Location):

            if self.world.in_bund([new_state]) and self.world.is_obstical([tuple(new_state)]) and not \
                    np.all(new_state == old_state.Location):

                if (self.goal_test(new_state)):
                    print('start ->', self.start, 'goal ->', self.goal,end='')
                    return old_state, old_state.cost + np.linalg.norm(new_state - old_state.Location)

                new_cost = old_state.cost + np.linalg.norm(new_state - old_state.Location)
                if not new_state.__str__() in self.open_list_dic:
                    if not new_state.__str__() in self.close_list_dic:
                        heuristic = self.get_heuristic(new_state)
                        new_node = Node(new_state, new_cost, heuristic)

                        self.insert_to_open_list(new_node, old_state)

                    else:
                        if old_state.cost < self.close_list_dic[new_state.__str__()].cost:
                            self.close_list_dic[new_state.__str__()] = old_state
                else:
                    if old_state.cost < self.open_list_dic[new_state.__str__()].cost:
                        self.open_list_dic[new_state.__str__()] = old_state

        return False, 0

    def goal_test(self, state):
        if np.all(state == self.goal):
            return True
        return False


class Node:

    def __init__(self, Location, cost=0, heuristics=0):
        self.Location = Location
        self.cost = cost
        self.heuristics = heuristics
