__author__ = 'omohamme'
import xml.etree.ElementTree as ET
import sys
sys.argv.pop(0)

for i in sys.argv:
    pareto_data = []
    pareto_observations = []
    tree = ET.parse(i)
    root = tree.getroot()
    for data_block in root.iter('_data'):
        pareto_points = []
        for data_item in data_block.iter("item"):
            # pareto_points.append(str(float(data_item.text)*3.14))
            pareto_points.append(str(float(data_item.text)))
        pareto_data.append(pareto_points)

    for data_block in root.iter('_objs'):
        pareto_points = []
        for data_item in data_block.iter("item"):
            pareto_points.append(data_item.text)
        pareto_observations.append(pareto_points)

    with open("pareto_clean.txt", "w") as file:
        final_hash = {}
        for item_index in range(len(pareto_data)):
            pareto_observations[item_index].reverse()
            final_hash[",".join(pareto_observations[item_index])] = ",".join(pareto_data[item_index])
        for item in final_hash:
            print >> file, final_hash[item], "--->", item
        print len(final_hash.keys())