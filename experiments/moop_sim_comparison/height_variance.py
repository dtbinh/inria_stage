__author__ = 'omohamme'

import math

height = open("height.txt","r")
height_lines = height.readlines()
height.close()

height_data = []
for line in height_lines:
    height_data.append(float(line))

# get the mean of the height
mean = 0.0
for item in height_data:
    mean = abs(item) + mean

mean = mean / len(height_data)

variance = 0.0
for item in height_data:
    variance = variance + (item - mean)**2

variance = variance / (len(height_data) - 1)

print mean
print variance