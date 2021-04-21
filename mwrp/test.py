# Floyd Warshall Algorithm in python
import numpy as np

# The number of vertices


class Floyd_warshall:

    def __init__(self,grid_map):
        self.grid_map=grid_map
        self.row=grid_map.shape[0]
        self.col=grid_map.shape[1]
        self.nV = grid_map.shape[0]**2
        self.dict_dist=dict()
        self.INF= (self.row*self.col)**2


    def from_grid_map_to_cost_map(self):
        cost_map=[]

        for i in range(self.row * self.col):

            tmp = [self.INF] * 16
            if self.grid_map[int(i / self.row)][int(i % self.row)] == 1:
                cost_map.append(tmp)
                continue
            if self.grid_map[int(i / self.row)][int(i % self.row)] == 0:
                tmp[i] = 0
            if i < self.row * self.col - 1 and int(i % self.row) + 1 < self.row and self.grid_map[int(i / self.row)][int(i % self.row) + 1] == 0:
                tmp[i + 1] = 1
            if i > 0 and i % self.row and self.grid_map[int(i / self.row)][int(i % self.row) - 1] == 0:
                tmp[i - 1] = 1
            if i + self.row < self.row * self.col and int(i / self.row) + 1 < self.row and self.grid_map[int(i / self.row) + 1][int(i % self.row)] == 0:
                tmp[i + self.row] = 1
            if i - self.row >= 0 and grid[int(i / self.row) - 1][int(i % self.row)] == 0:
                tmp[i - self.row] = 1
            cost_map.append(tmp)
        return cost_map

    # Algorithm implementation
    def floyd_warshall(self):
        cost_map=self.from_grid_map_to_cost_map()

        distance = list(map(lambda i: list(map(lambda j: j, i)),cost_map))

        # Adding vertices individually
        for k in range(self.nV):
            for i in range(self.nV):
                for j in range(self.nV):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

        for ii in range(len(distance)):
            for jj in range(len(distance)):
                cell_a=tuple((int(jj / self.row),int(jj % self.row)))
                cell_b=tuple((int(ii / self.row),int(ii % self.row)))
                if not distance[ii][jj] == self.INF:
                    key=tuple(sorted([cell_a,cell_b]))
                    if key not in self.dict_dist:
                        self.dict_dist[key]=distance[ii][jj]
#         x=1
#
#
#
# grid=[[0,0,0,0],
#           [0,0,0,0],
#           [1,1,1,0],
#           [0,0,0,0]]
#
#
#
#
# a=Floyd_warshall(np.array(grid))
# a.floyd_warshall()