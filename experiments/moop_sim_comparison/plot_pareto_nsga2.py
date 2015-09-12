__author__ = 'omohamme'
from pylab import *
from PyGMO import *
import sys
import numpy
from PyGMO import *

def pareto_frontier(Xs, Ys, maxX = False, maxY = False):
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    p_front = [myList[0]]
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

sys.argv.pop(0)

for i in sys.argv:
    data = np.loadtxt(i)
    print len(data)
    try:
        clean_x, clean_y = pareto_frontier(-numpy.array(data[:, 2]), -numpy.array(data[:, 3]))
    except:
        clean_x, clean_y = pareto_frontier(-numpy.array(data[:, 0]), -numpy.array(data[:, 1]))

    plot(clean_x, clean_y, 'o', label=i, marker='o', linestyle='--')

    clean_data = []
    for i in range(len(clean_x)):
        clean_data.append((clean_x[i], clean_y[i]))
    hv = util.hypervolume(clean_data)
    ref_point = (2, 2) # x is the 1-speed, y is the variance in height
    print i, " -- > ", str(hv.compute(r=ref_point))
legend(loc=4)
show()