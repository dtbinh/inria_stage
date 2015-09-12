__author__ = 'omohamme'
import os

for filename in os.listdir("."):
    if "ehvi_100gp" in filename:
        names = filename.split("ehvi_100gp")
        names.insert(1,"ehvi100gp")
        new_name = "".join(names)
        print filename
        print new_name
        os.rename(filename, new_name)