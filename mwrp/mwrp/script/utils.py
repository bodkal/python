import numpy as np
from random import randint
from time import time
import matplotlib.pyplot as plt
import csv


class Utils:

    @staticmethod
    def n_dim_distance(point_a, point_b):
        new_vec = point_a - point_b
        return np.sqrt(np.dot(new_vec, new_vec))

    @staticmethod
    def distance_all(point_a, points):
        new_vec = points - point_a
        return np.sqrt(np.einsum('ij,ij->i', new_vec, new_vec))
        # return cdist([point_a], points)

    @staticmethod
    def get_random_map(row, col, start):
        random_map = np.zeros((row, col))
        random_map[row - 1, :] = 1
        random_map[0, :] = 1
        random_map[:, col - 1] = 1
        random_map[:, 0] = 1
        for i in range(int(row * col / 10)):
            random_map[randint(1, row - 1), randint(1, col - 1)] = 1

        for i in start:
            random_map[i] = 0
        return random_map

    @staticmethod
    def centrality_list(dist_dict, grid_map):
        centrality_list = []
        for index, cell in enumerate(dist_dict):
            tmp_cell_all_dist = sum(cell)
            if tmp_cell_all_dist:
                centrality_list.append(tuple((tmp_cell_all_dist, tuple((divmod(index, grid_map.shape[1]))))))
        centrality_list = sorted(centrality_list, reverse=True)
        return centrality_list

    @staticmethod
    def centrality_dict(dist_dict, dist_map):
        centrality_dict = dict()
        for index, cell in enumerate(dist_dict):
            #tmp_cell_all_dist = sum(cell)
            if cell:
                centrality_dict[dist_map[index]] = cell
        return centrality_dict

    @staticmethod
    def centrality_dict_wachers(dist_dict, dist_map):
        centrality_dict = dict()
        for index, cell in enumerate(dist_dict):
            tmp_cell_all_dist = sum(cell)
            if tmp_cell_all_dist:
                centrality_dict[dist_map[index]] = tmp_cell_all_dist
        return centrality_dict

    @staticmethod
    def centrality_tie_see(dist_dict, grid_map, dist_wachers):
        centrality_list = []
        for index, cell in enumerate(dist_dict):
            tmp_cell_all_dist = sum(cell)
            tmp_index = 0
            if tmp_cell_all_dist:
                new_cell = tuple((divmod(index, grid_map.shape[1])))
                for i in range(centrality_list.__len__()):
                    if tmp_cell_all_dist > centrality_list[i][0]:
                        break
                    elif tmp_cell_all_dist == centrality_list[i][0]:
                        if dist_wachers[new_cell].__len__() < dist_wachers[centrality_list[i][1]].__len__():
                            break
                    tmp_index += 1

                centrality_list.insert(tmp_index, tuple((tmp_cell_all_dist, new_cell)))
        return centrality_list

    @staticmethod
    def centrality_tie_wahcers(dist_dict, grid_map, dist_wachers):
        centrality_list = []
        for index, cell in enumerate(dist_dict):
            tmp_cell_all_dist = sum(cell)
            tmp_index = 0
            if tmp_cell_all_dist:
                new_cell = tuple((divmod(index, grid_map.shape[1])))
                for i in range(centrality_list.__len__()):
                    if dist_wachers[new_cell].__len__() < dist_wachers[centrality_list[i][1]].__len__():
                        break
                    elif dist_wachers[new_cell].__len__() < dist_wachers[centrality_list[i][1]].__len__():
                        if tmp_cell_all_dist > centrality_list[i][0]:
                            break
                    tmp_index += 1

                centrality_list.insert(tmp_index, tuple((tmp_cell_all_dist, new_cell)))
        return centrality_list


    @staticmethod
    def centrality_meen_see(dist_wachers):
        centrality_list = []
        for cell in dist_wachers:
            centrality_list.append(tuple((dist_wachers[cell].__len__(), cell)))
        centrality_list = sorted(centrality_list)
        return centrality_list

    @staticmethod
    def print_pivot(world,pivot):
        tmp = np.copy(world.grid_map)
        for cell in pivot.keys():
            tmp[cell] = 3
        #plt.figure(1)
        plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
       # plt.show()

    @staticmethod
    def print_whacer(world, tmp_cell):
        for pivot_cell in tmp_cell:
            tmp = np.copy(world.grid_map)
            for cell in world.dict_wachers[pivot_cell]:
                tmp[cell] = 3
            tmp[pivot_cell] = 2

           # plt.figure(pivot_cell.__str__())
            plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
            plt.gca().set_aspect('equal')
            plt.gca().set_ylim(plt.gca().get_ylim()[::-1])

        plt.show()

    @staticmethod
    def print_all_whacers(world, tmp_cell):
        tmp = np.copy(world.grid_map)

        for pivot_cell in tmp_cell:
            for cell_2 in pivot_cell:
                for cell in world.dict_wachers[cell_2]:
                    if not tmp[cell]==2:
                        tmp[cell] = 3
                tmp[cell_2] = 2


            plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
            plt.gca().set_aspect('equal')
            plt.gca().set_ylim(plt.gca().get_ylim()[::-1])

        plt.show()

    @staticmethod
    def print_fov(grid_map,all_cell,main_cell):
        tmp = np.copy(grid_map)

        for cell in all_cell:
            if not tmp[cell]==2:
                tmp[cell] = 3
            tmp[main_cell] = 2

        plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
        plt.show()


    @staticmethod
    def map_to_sets(cell):
        return set(map(tuple,[cell]))

    @staticmethod
    def convert_map(map_config):
        #row_map = []

        with open(map_config, newline='') as txtfile:
            #all_row = []
            row_map=[ [1 if cell == '@' else 0 for cell in row[:-1]] for row in txtfile.readlines()]

        return row_map

class Node:

    def __init__(self, parent, location, unseen, cost=0, heuristics=0):
        self.parent = parent
        self.location = location
        self.unseen = unseen
        self.cost = cost
        self.heuristics = heuristics

    def __sort__(self):
        return tuple(sorted((self.location)))

class Bresenhams:
    def __init__(self, grid_map):
        self.grid_map = grid_map

        self.initial = True
        self.end = True
        self.x0 = 0
        self.x1 = 0
        self.y0 = 0
        self.y1 = 0
        self.dx = 0
        self.dy = 0
        self.sx = 0
        self.sy = 0
        self.err = 0

    def get_next(self):
        if self.initial:
            self.initial = False
            return tuple((self.x0, self.y0))
        if self.x0 == self.x1 and self.y0 == self.y1 or self.grid_map[self.x0, self.y0] == 1:
            self.end = True
            return False

        e2 = 1 * self.err
        if e2 > -self.dy:
            self.err = self.err - self.dy
            self.x0 = self.x0 + self.sx

        if e2 < self.dx:
            self.err = self.err + self.dx
            self.y0 = self.y0 + self.sy

        return tuple((self.x0, self.y0))

    def get_line(self, start_cell, end_cell):
        self.initial = True
        self.end = False
        self.x0 = start_cell[0]
        self.y0 = start_cell[1]
        self.x1 = end_cell[0]
        self.y1 = end_cell[1]

        self.dx = abs(self.x1 - self.x0)
        self.dy = abs(self.y1 - self.y0)

        self.sx = 1 if self.x0 < self.x1 else -1
        self.sy = 1 if self.y0 < self.y1 else -1
        self.err = self.dx - self.dy

        all_seen = []
        while not self.end:
            new_seen = self.get_next()
            if new_seen:
                all_seen.append(new_seen)
        return set(map(tuple, all_seen[:-1]))


class FloydWarshall:

    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.free_cell = np.array(np.where(self.grid_map == 0)).T

        self.row = grid_map.shape[0]
        self.col = grid_map.shape[1]
        self.nV = self.free_cell.__len__()
        self.dict_dist = dict()
        self.INF = (self.row * self.col) * 2

    def from_grid_map_to_cost_map(self):

        cost_map = []
        for main_cell in self.free_cell:
            cost_map.append([abs(x[0] - main_cell[0]) + abs(x[1] - main_cell[1]) if abs(x[0] - main_cell[0]) + abs(
                x[1] - main_cell[1]) < 2 else self.INF for x in self.free_cell])

        return cost_map

    def floyd_warshall(self):
        cost_map = self.from_grid_map_to_cost_map()
        for k in range(self.nV):
            for i in range(self.nV):
                for j in range(self.nV):
                    cost_map[i][j] = min(cost_map[i][j], cost_map[i][k] + cost_map[k][j])

        for ii in range(self.nV):
            for jj in range(self.nV):

                cell_a = tuple(self.free_cell[ii])
                cell_b = tuple(self.free_cell[jj])
                key = tuple(sorted([cell_a, cell_b]))
                if key not in self.dict_dist:
                    self.dict_dist[key] = cost_map[ii][jj]

        centrality_dict = {tuple(self.free_cell[index]) : sum(row) for index, row in enumerate(cost_map)}
        return self.dict_dist, centrality_dict


from docplex.mp.model import Model
import docplex.mp.solution as mp_sol

class lp_mtsp():

    def __init__(self,agent_manber):#,agent_locashon,distance_dict):
        self.mdl = Model('SD-MTSP')
        self.m = agent_manber

        # self.mdl = Model('SD-MTSP')
        #
        # distance_out_agent={(0,i) : 0 for i in range(1,self.m+1)}
        # distance_in_agent={(i,0) : 0 for i in range(1,self.m+1)}
        #
        # self.permanent_distance={**distance_out_agent,**distance_in_agent}
        #
        # self.x_permanent = self.mdl.binary_var_dict(self.permanent_distance, name='x')
        # self.u_permanent = self.mdl.continuous_var_list(range(self.m+1),lb=0, name='u')
        #
        # for i in range(1,self.m+1):
        #     self.mdl.add_constraint(self.x_permanent[(0, i)]  == 1, ctname=f'in_start_{i}')


        #print(self.mdl.export_to_string())




    def get_sum_of_cost(self):
       # L = 70


        ind = [0] * self.citys[1:].__len__()

        #all_cell_location_no_start = [(i, j) for i in self.citys for j in self.citys if i != j and i != 0 and j != 0]
        all_cell_location = [(i, j) for i in self.citys for j in self.citys if i != j]

        mdl = Model('SD-MTSP')

        x = mdl.binary_var_dict(all_cell_location, name='x')
        u = mdl.continuous_var_list(self.citys, name='u')

        mdl.minimize(mdl.sum(self.distance_dict[e] * x[e] for e in all_cell_location))

       # mdl.minimize(mdl.sum(u[i] * x[(i,0)] for i in self.citys[1:] if (i,0) in x))

        for c in self.citys[1:]:
            mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in all_cell_location if i == c) == 1, ctname='out_%d' % c)
            mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in all_cell_location if j == c) == 1, ctname='in_%d' % c)

        for c in self.agent_locashon:
            mdl.add_constraint(x[(0, c)] == 1, ctname='out_dipot_%d' % c)

        for c in self.agent_locashon:
            mdl.add_constraint(x[(0, c)] == 1, ctname='out_dipot_%d' % c)

        # mdl.add_constraint(mdl.sum(x[(0,j)] for j in range(n) if j != 0) == m, ctname='out_start')
        mdl.add_constraint(mdl.sum(x[(j, 0)] for j in self.citys[1:] if j != 0) == self.m, ctname='in_start')

        for i, j in all_cell_location:
            if j != 0:
                mdl.add_indicator(x[(i, j)], u[j] - u[i] >= self.distance_dict[(i, j)], name='order_(%d,_%d)' % (i, j))

        # for i in self.citys[1:]:
        #     ind[i - self.citys[1]] = mdl.add_indicator(x[(i, 0)], u[i] <= L, name=f'HB_2_{i}')

        solucion = mdl.solve(log_output=False)

        return solucion.objective_value/self.m

    def get_makespan1(self,for_plot,w,p,n,tmp_distance):

        #self.agent_locashon = agent_locashon
        #self.citys = citys

        tmp_mdl=self.mdl.clone()

        per_var=[i for i in tmp_mdl.iter_variables()]

            #.clone(new_name='tmp_model')

        citys=range(self.m+self.u_permanent.__len__())
        ind = [0] * citys[1:].__len__()
        distance_dict={**self.permanent_distance,**tmp_distance}
        all_cell_location = list(distance_dict.keys())

        x_tmp = tmp_mdl.binary_var_dict(tmp_distance, name='x')
        u_tmp = tmp_mdl.continuous_var_list(citys[self.m+1:], name='u')

        x = {**x_tmp, **dict(zip(self.x_permanent.keys(), per_var[:self.x_permanent.__len__()])) }
        u=per_var[self.x_permanent.__len__():]+u_tmp

        tmp_mdl.minimize(tmp_mdl.sum(distance_dict[e] * x[e] for e in all_cell_location))

        for c in citys[1:]:
            tmp_mdl.add_constraint( tmp_mdl.sum(x[(i, j)] for i, j in all_cell_location if i == c) == 1, ctname='out_%d' % c)

        for c in citys[self.m+1:]:
            tmp_mdl.add_constraint(tmp_mdl.sum(x[(i, j)] for i, j in all_cell_location if j == c ) == 1, ctname='in_%d' % c)

        tmp_mdl.add_constraint(tmp_mdl.sum(x[(j, 0)] for j in citys[1:] if j != 0) == self.m, ctname='in_start')

        for i, j in all_cell_location:
            if distance_dict[(i, j)]:
                tmp_mdl.add_indicator(x[(i, j)], u[j] == u[i] + distance_dict[(i, j)], name='order_(%d,_%d)' % (i, j))

        print(tmp_mdl.export_to_string())
        solucion = tmp_mdl.solve(log_output=False)

        max_u = self.get_subtoor(x, all_cell_location,distance_dict)
        #self.print_SD_MTSP_on_map(for_plot, all_cell_location, x, w, p,citys)

        while solucion:
            L = int(max(max_u) - 1)
            for i in citys[1:]:
                ind[i - citys[1]] = tmp_mdl.add_indicator(x[(i, 0)], u[i] <= L, name=f'HB_2_{i}')
            solucion = tmp_mdl.solve(log_output=False)

            if solucion:
                tmp_mdl.remove_constraints(ind)
                max_u = self.get_subtoor(x, all_cell_location, distance_dict)
                #self.print_SD_MTSP_on_map(for_plot, all_cell_location, x, w, p, citys)
                xx=1

        return max(max_u)

    def get_makespan(self, for_plot, w, p, n, citys,distance_dict):

        #lp_mtsp.get_makespan(for_plot, self.world, pivot, self.number_of_node, citys)

        #self.agent_locashon = agent_locashon
        #self.citys = citys
        self.mdl.clear()
        self.mdl.set_time_limit(1)
        self.distance_dict = distance_dict

        self.citys=citys
        ind = [0] * self.citys[1:].__len__()
        all_cell_location = list(self.distance_dict.keys())


        x = self.mdl.binary_var_dict(all_cell_location, name='x')
        u = self.mdl.continuous_var_list(self.citys, name='u')

        self.mdl.minimize(self.mdl.sum(self.distance_dict[e] * x[e] for e in all_cell_location))

        for c in self.citys[1:]:
            self.mdl.add_constraint(self.mdl.sum(x[(i, j)] for i, j in all_cell_location if i == c) == 1, ctname='out_%d' % c)
            self.mdl.add_constraint(self.mdl.sum(x[(i, j)] for i, j in all_cell_location if j == c) == 1, ctname='in_%d' % c)

        self.mdl.add_constraint(self.mdl.sum(x[(0,j)] for j in self.citys[1:self.m+1]  if j != 0) == self.m, ctname='out_start')
        self.mdl.add_constraint(self.mdl.sum(x[(j, 0)] for j in self.citys[1:] if j != 0) == self.m, ctname='in_start')

        for i, j in all_cell_location:
            if self.distance_dict[(i, j)]:
                self.mdl.add_indicator(x[(i, j)], u[j] == u[i] + self.distance_dict[(i, j)], name='order_(%d,_%d)' % (i, j))
               # mdl.add_indicator(x[(i, j)], u[j] >= u[i] + self.distance_dict[(i, j)], name='order_(%d,_%d)' % (i, j))

        solucion = self.mdl.solve(log_output=False)
        if self.mdl.solve_status.name != 'OPTIMAL_SOLUTION':
            print('-1')
            return -1
        #print(solucion.solve_details)
        #print(solucion.display())

        #max_u = solucion.get_all_values()[-self.citys.__len__():]

        #arcos_activos0,arcos_activos1 = [e for e in all_cell_location if x[e].solution_value > 0.9]
        #return max(max_u)
        #print(f' min_sum  {solucion.objective_value} HB : {L}  min_max : {max(max_u)}')

        #self.print_SD_MTSP_on_map(for_plot, all_cell_location, x, w, p)
        max_u = self.get_subtoor(x, all_cell_location,self.distance_dict)

        #print(f'n : {n} HB : {val}  min_max : {max_u}')

        #print(solucion.display())
        #self.print_SD_MTSP_on_map(for_plot, all_cell_location, x, w, p)

        while solucion:

            L = int(max(max_u) - 1)
            #mdl.change_var_upper_bounds(u,L)
            for i in self.citys[1:]:
                ind[i - self.citys[1]] = self.mdl.add_indicator(x[(i, 0)], u[i] <= L, name=f'HB_2_{i}')
            solucion = self.mdl.solve(log_output=False)


            if solucion:
                self.mdl.remove_constraints(ind)
                max_u = self.get_subtoor(x,all_cell_location,self.distance_dict)
                if self.mdl.solve_status.name!='OPTIMAL_SOLUTION':
                    print('-1')
                    return -1

                # solucion.get_all_values()[-self.citys.__len__():]
                # print(f'n : {n} HB : {val}  min_max : {max_u}')
                # print(f' min_sum  {solucion.objective_value} HB : {L}  min_max : {max(max_u)}')
                # print(solucion.display())
                # print(mdl.export_to_string())
                # self.print_SD_MTSP_on_map(for_plot, all_cell_location, x, w, p)

        #print(f'pivot = {p.__len__()}',end='\t')
        return max(max_u)

    def print_SD_MTSP_on_map(self,for_plot,all_cell_location,x,w,p):

        import matplotlib.pyplot as plt
        plt.figure('0')
        arcos_activos = [e for e in all_cell_location if x[e].solution_value > 0.9]
        for i, j in arcos_activos:
            plt.plot([for_plot[i][1]+0.5, for_plot[j][1]+0.5], [for_plot[i][0]+0.5, for_plot[j][0]+0.5], color='r', alpha=0.6,linewidth=2, marker='o')
        # for i, j in for_plot:
        #     plt.scatter(j, i, color='r', zorder=1)
        for i in self.citys:
            plt.annotate(i, xy=(for_plot[i][1]+0.5, for_plot[i][0]+0.5),color ='k',weight='bold')
        Utils.print_pivot(w,p)
        plt.show()

    def get_subtoor(self,x,all_cell_location,distance_dict):
        ac0 = []
        ac1 = []
        for e in all_cell_location:
            if x[e].solution_value > 0.9:
                ac0.append(e[0])
                ac1.append(e[1])

        val = []
        while ac1.__len__():
            tmp_x = ac1.pop(0)
            del ac0[0]
            val.append(0)

            while tmp_x and tmp_x in ac0:
                i = ac0.index(tmp_x)
                val[-1] += distance_dict[(ac0[i], ac1[i])]

                tmp_x = ac1[i]
                del ac0[i]
                del ac1[i]
        return val