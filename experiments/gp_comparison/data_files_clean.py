__author__ = 'omohamme'

for i in [1]:
    with open("all_outputs_"+str(i)+"d.txt", "r") as file:
        file_lines = file.readlines()

    data = {}
    print len(file_lines)

    for line in file_lines:
        value, key = line.split("--->")
        data[key] = value

    print len(data.keys())
    with open("all_outputs_"+str(i)+"d_x.txt", "w") as file:
        for item in data:
            print >> file, data[item] + "--->" + item