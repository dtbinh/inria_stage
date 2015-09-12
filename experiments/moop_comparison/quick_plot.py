#Quick and dirty plot func
import numpy as np;

import matplotlib.pyplot as plt;

#t = np.arange(0., 5., 0.2)

## red dashes, blue squares and green triangles
#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
#plt.show()
def str_to_float (numbers):
    print numbers
    return float(numbers)

with open("ehvi.txt","r") as file:
    ehvi_lines = file.readlines()

with open("nsbo.txt","r") as file:
    nsbo_lines = file.readlines()

with open("nsga2.txt","r") as file:
    nsga2_lines = file.readlines()

with open("parego.txt","r") as file:
    parego_lines = file.readlines()
    
ehvi_results_x = []
ehvi_results_y = []
for line in ehvi_lines:
    #if (float(line.split(",")[0]) >= 0) and (float(line.split(",")[1]) >= 0):
    ehvi_results_x.append(-float(line.split(",")[0]))
    ehvi_results_y.append(-float(line.split(",")[1]))
    
nsbo_results_x = []
nsbo_results_y = []
for line in nsbo_lines:
    #if (float(line.split(",")[0]) >= 0) and (float(line.split(",")[1]) >= 0):
    nsbo_results_x.append(-float(line.split(",")[0]))
    nsbo_results_y.append(-float(line.split(",")[1]))

nsga2_results_x = []
nsga2_results_y = []
for line in nsga2_lines:
    #print line
    #if (float(line.split(",")[0]) >= 0) and (float(line.split(",")[1]) >= 0):
    nsga2_results_x.append(-float(line.split(",")[0]))
    nsga2_results_y.append(-float(line.split(",")[1]))
    
parego_results_x = []
parego_results_y = []
for line in parego_lines:
    #if (float(line.split(",")[0]) >= 0) and (float(line.split(",")[1]) >= 0):
    parego_results_x.append(-float(line.split(",")[0]))
    parego_results_y.append(-float(line.split(",")[1]))

optimal_front = []
#with open("dtlz7.txt","r") as file:
with open("zdt2.txt","r") as file:
    optimal_front = file.readlines()
    
optimal_results_x = []
optimal_results_y = []
for line in optimal_front:
    optimal_results_x.append(float(line.split(",")[0]))
    optimal_results_y.append(float(line.split(",")[1]))
    
with open("sferes_zdt2.py","r") as file:
    sferes2_front = file.readlines()
    
sferes2_results_x = []
sferes2_results_y = []
for line in sferes2_front:
    #print line
    sferes2_results_x.append(-float(line.split(" ")[2]))
    sferes2_results_y.append(-float(line.split(" ")[3]))
    
print "len(parego_results_x) = ",len(parego_results_x)
plt.figure(1)
#plt.plot(ehvi_results_x,ehvi_results_y,'bo',nsbo_results_x,nsbo_results_y,'gs',nsga2_results_x,nsga2_results_y,'r^',parego_results_x,parego_results_y,'b+',optimal_results_x,optimal_results_y,"r+")
plt.plot(ehvi_results_x,ehvi_results_y,'bo',nsga2_results_x,nsga2_results_y,'r^',parego_results_x,parego_results_y,'b+',optimal_results_x,optimal_results_y,"r+")
plt.legend(["EHVI","NSGA2","PAREGO","OPTIMAL"])
plt.figure(2)
plt.plot(nsbo_results_x,nsbo_results_y,'bo')
plt.legend(["NSBO"])
#plt.plot(ehvi_results_x,ehvi_results_y,'bo',nsbo_results_x,nsbo_results_y,'gs',nsga2_results_x,nsga2_results_y,'r^',parego_results_x,parego_results_y,'b+')
#plt.figure(1)
#plt.plot(nsga2_results_x,nsga2_results_y,"bo",optimal_results_x,optimal_results_y,"r+")
#plt.figure(2)
#plt.plot(nsga2_results_x,nsga2_results_y,"bo")
#plt.figure(3)
#plt.plot(nsbo_results_x,nsbo_results_y,"bo")
#plt.legend(["SFERES-NSGA2","OPTIMAL"])
#plt.legend(["EHVI","NSBO","NSGA2","PAREGO"])
plt.show()