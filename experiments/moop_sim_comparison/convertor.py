__author__ = 'omohamme'
import sys

if sys.argv[1][-1] == ",":
    argument = sys.argv[1].split(",")[:-1]
else:
    argument = sys.argv[1].split(",")

output = ""
for para in argument:
    # phase = float(para)*(3.14/2.0)
    phase = float(para)*(3.14)
    output = output + str(phase) + ","
print output

