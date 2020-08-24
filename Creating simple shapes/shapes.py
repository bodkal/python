import numpy as np
from numpy.linalg import lstsq
from numpy import ones,vstack
import math
import matplotlib.pyplot as plt


class shapes:
    def __init__(self):
        print(
            "for Circle -> Circle(xcenter,ycenter,raduse,number of points)\nfor polygon->polygon(np.array([[x1,y1],[x2,y2],...]),Number of points per side)\n for line -> line(np.array([[x1,y1],[x2,y2]),number og points])")

    def coaff_line(self, points):
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        return np.round_(m, 5), np.round_(c, 5)

    def get_x(self,data,NOP):
        if (data[0, 0] < data[1, 0]):
            x = np.arange(data[0, 0], data[1, 0], math.fabs((data[1, 0] - data[0, 0]) / NOP))
        else:
            x = np.arange(data[1, 0], data[0, 0], math.fabs((data[1, 0] - data[0, 0]) / NOP))
        return x

    def circle(self, x, y, r, NOP):
        points = np.array([[0, 0]])
        for i in np.arange(0, 2 * math.pi, (2 * math.pi) / NOP):
            points = np.append(points.T, np.array([[x + r * math.cos(i), y + r * math.sin(i)]]).T, axis=1).T
        points = np.delete(points, 0, 0)
        return np.round_(points, 5)

    def polygon(self, data, NOP):
        points = np.array([[0, 0]])
        for i in range(len(data) - 1):
            if(data[i+1,0]-data[i,0]==0):
                x = np.ones(NOP) * data[i, 0]
                y = self.get_x(np.array([[data[i+1, 1], data[i+1, 0]], [data[i, 1], data[i, 0]]]), NOP)
            else:
                m, c = self.coaff_line([(data[i, 0], data[i, 1]), (data[i + 1, 0], data[i + 1, 1])])
                x = self.get_x(np.array([data[i], data[i+1]]),NOP)
                y = m * x + c
            points = np.append(points.T, np.array([x, y]), axis=1).T

        if (data[-1, 0] - data[0, 0] == 0):
            x = np.ones(NOP) * data[-1, 0]
            y = self.get_x(np.array([[data[-1, 1], data[-1, 0]], [data[0, 1], data[0, 0]]]), NOP)
        else:
            m, c = self.coaff_line([(data[-1, 0], data[-1, 1]), (data[0, 0], data[0, 1])])
            x=self.get_x(np.array([data[-1], data[0]]),NOP)
            y = m * x + c
        points = np.append(points.T, np.array([x, y]), axis=1).T
        points = np.delete(points, 0, 0)
        return np.round_(points, 5)


    def line(self, data, NOP):
        if(data[1,0]-data[0,0]==0):
            x=np.ones(NOP)*data[1,0]
            y=self.get_x(np.array([[data[1,1],data[1,0]],[data[0,1],data[0,0]]]),NOP)
        else:
            m, c = self.coaff_line([(data[0, 0], data[0, 1]), (data[1, 0], data[1, 1])])
            x=self.get_x(data,NOP)
            y=m*x+c
        return np.round_(np.array([x, y]), 5).T


if __name__ == '__main__':
    a=shapes()
    b=a.line(np.array([[6,6],[6,1]]),100)
    for i in range(len(b)):
        plt.plot(b[i,0],b[i,1],'ob')
    plt.show()