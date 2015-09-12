__author__ = 'omohamme'

import random

for i in range(1000000):
    # print i
    x = []
    for j in range(5):
        # x.append(1)
        x.append(random.random())

    for item1_index in range(len(x)):
        for item2_index in range(len(x)):
            if item1_index != item2_index:
                if x[item1_index] == x[item2_index]:
                    print "ERROR---->"
                    print "The whole set = ", x
                    print "Item indexes = ", item1_index, ", ", item2_index
                    print "---------------------------------------------------------------"
