import numpy as np
from script.utils import Node, Utils
from script.world import WorldMap
from treelib import Tree
# need to chack it -> from scipy.spatial import KDTree
from time import time
from scipy.spatial.distance import cdist


class Mwrp:
    def __init__(self, world, number_of_agent, start_node):
        self.number_of_agent = number_of_agent
        self.open_list = np.array([start_node])
        self.tree = Tree()
        self.tree.create_node(start_node.pos.__str__(), start_node.pos.__str__(), data=start_node)
        self.world = world
        self.node_expend_index = 1
        self.need_to_see=np.sum(self.world.grid_map==0)


    def insert_to_open_list(self, new_node):
        for index, data in enumerate(self.open_list):
            if data.f > new_node.f:
                self.open_list = np.insert(self.open_list, index, new_node)
                return
        self.open_list = np.append(self.open_list, new_node)

    def pop_open_list(self):
        return self.open_list[0].pos

    def move_from_open_to_close(self, index=0):
        self.open_list = np.delete(self.open_list, index)

    def heuristic(self, state):
        return 1

    # #old
    # def fix_g_subtree(self,neighbor,state):
    #     tmp_new_open = []
    #     #self.tree.move_node(neighbor.__str__(), state.__str__())
    #
    #     for node in mwrp.tree.subtree(state.data.pos.__str__()).expand_tree(mode=Tree.DEPTH):
    #         parent = mwrp.tree.parent(node)
    #
    #         new_g = parent.data.g + Utils.n_dim_distance(
    #             self.tree.get_node(node).data.pos, parent.data.pos)
    #         self.tree.get_node(node).data.f = self.tree.get_node(node).data.f - self.tree.get_node(node).data.g + new_g
    #
    #         self.tree.get_node(node).data.g = new_g
    #         tmp_new_open.append(self.tree.get_node(node))
    #
    #     for need_new_open in tmp_new_open:
    #         for index, data in enumerate(self.open_list):
    #             if (np.all(data.pos == need_new_open.data.pos)):
    #                 self.open_list = np.delete(self.open_list, index)
    #                 self.insert_to_open_list(self.tree.get_node(need_new_open.data.pos.__str__()).data)
    #                 break

    def get_all_seen(self, state):

        tmp_node = self.tree.get_node(state.identifier)
        tmp_seen = self.world.get_seen(tmp_node)

        while not self.tree.get_node(tmp_node.identifier).is_root():
            tmp_node = self.tree.get_node(tmp_node.predecessor(self.tree.identifier))
            tmp_seen = np.vstack((tmp_seen, self.world.get_seen(tmp_node)))

        all_seen = np.unique(tmp_seen, axis=0)

        return all_seen

    # def get_all_seen1(self, state):
    #
    #     tmp_node = self.tree.get_node(state.identifier)
    #     dictOfWords = {i.__str__(): 0 for i in self.world.get_seen(tmp_node)}
    #
    #     while not self.tree.get_node(tmp_node.identifier).is_root():
    #         tmp_node = self.tree.get_node(tmp_node.predecessor(self.tree.identifier))
    #         for seen in self.world.get_seen(tmp_node):
    #             if not seen.__str__() in dictOfWords:
    #                 dictOfWords[seen.__str__()]=0
    #
    #     return dictOfWords

    # def expend_all(self,state):
    #     for neighbor in self.world.get_neighbors(state):
    #         if self.world.in_bund(neighbor) and self.world.is_obstical(neighbor):
    #             new_g = mwrp.tree.get_node(state.__str__()).data.g + Utils.n_dim_distance(state, neighbor)
    #             if not self.tree.get_node(neighbor.__str__()):
    #                 h = self.heuristic(state)
    #                 new_node=Node(neighbor,new_g,new_g+h)
    #                 self.tree.create_node(neighbor.__str__(),neighbor.__str__(),parent=(state.__str__()),data=new_node)
    #                 self.insert_to_open_list(new_node)
    #             else:
    #                 self.fix_g(neighbor, state)
    #
    #     #self.tree.show()
    #     self.move_from_open_to_close()

    def expend(self, state):
        move_index = np.zeros(self.number_of_agent).astype(int)
        for i in range(LOS ** number_of_agent):
            for j in range(number_of_agent):
                i, index = divmod(i, LOS)
                move_index[j] = index
            neighbor = self.world.get_one_neighbor(state, move_index)

            if self.world.in_bund(neighbor) and self.world.is_obstical(neighbor):
                new_g = mwrp.tree.get_node(state.__str__()).data.g + Utils.n_dim_distance(state, neighbor)
                if not self.tree.contains(neighbor.__str__()):
                    h = self.heuristic(state)
                    new_node = Node(neighbor, new_g, new_g + h)
                    self.tree.create_node(neighbor.__str__(), neighbor.__str__(), parent=(state.__str__()),
                                          data=new_node)
                    self.insert_to_open_list(new_node)
                else:
                    if new_g < self.tree.get_node(neighbor.__str__()).data.g:
                        self.fix_g(neighbor, state)

        self.move_from_open_to_close()

    def fix_g(self, old_state, new_parent):
        state = self.tree.get_node(old_state.__str__())

        old_parent = self.tree.get_node(state.predecessor(self.tree.identifier))

        new_parent = self.tree.get_node(new_parent.__str__())

        if self.seen_comparison(new_parent, old_parent):

            self.tree.move_node(state.identifier, new_parent.identifier)

            new_g = new_parent.data.g + \
                    Utils.n_dim_distance(self.tree.get_node(state.identifier).data.pos, new_parent.data.pos)

            self.tree.get_node(state.identifier).data.f = self.tree.get_node(state.identifier).data.f - \
                                                          self.tree.get_node(state.identifier).data.g + new_g

            self.tree.get_node(state.identifier).data.g = new_g

            for index, data in enumerate(self.open_list):
                if np.all(data.pos == state.data.pos):
                    self.open_list = np.delete(self.open_list, index)
                    self.insert_to_open_list(self.tree.get_node(state.identifier).data)
                    break

    def goal_test(self,state):
        seen_number=self.world.get_seen(state).shape[0]
        if(seen_number==self.need_to_see):
            return True
        return False

    def seen_comparison(self, state_new, state_old):

        seen_new = self.get_all_seen(state_new).tolist()

        seen_old = self.get_all_seen(state_old).tolist()

        if seen_new.shape[0] < seen_old.shape[0]:
            return False
        for one_seen in seen_old:
            if one_seen not in seen_new:
                return False
        return True

    # def seen_comparison1(self, state_new, state_old):
    #     a = np.min(cdist(state_new, state_old, 'cityblock'), axis=0)
    #     if np.all(a):
    #         return False
    #     return True

    # def seen_comparison1(self, state_new, state_old):
    #     seen_new = self.get_all_seen1(state_new)
    #     seen_old = self.get_all_seen1(state_old)
    #     for i in seen_old.keys():
    #         if not i in seen_new:
    #             return False
    #     return True



if __name__ == '__main__':
    map_type = 'map_a'
    map_config = './config/{}_config.csv'.format(map_type)
    map = np.genfromtxt(map_config, delimiter=',', case_sensitive=True)

    number_of_agent = 2
    LOS = 4
    start_pos = np.ones(number_of_agent * 2).astype(int)
    world = WorldMap(map, number_of_agent, LOS)
    mwrp = Mwrp(world, number_of_agent, Node(start_pos, 0, 0))

    mwrp.expend(start_pos)
    print('----')
    mwrp.tree.show()
    mwrp.expend(start_pos + [1, 0, 1, 0])
    mwrp.goal_test(mwrp.tree.get_node('[3 1 3 1]'))
    print('----')

    mwrp.tree.show()

    # print(world.get_one_neighbor(start_pos,move_index))

# v=world.get_neighbors(start_pos)


# TO DO:
# whacers / los Bresenham ->
# heuristic ->
# find pivot ->

# improve g swich to seen chack -> V
# movment index  insted of a list -> V
# bild seen fun for all cell obove the tree -V
