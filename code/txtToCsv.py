from numpy import *
from pandas import *
from air_combat.code.const import *
import csv


class TxtToCsv():
    def convert(self):
        lines = []
        with open(DATAPATH, 'r') as f1:
            data = f1.readlines()
            for line in data:
                arr = line.split(';')
                #print(arr)
                arr_last = arr.pop(-1)
                #print(arr_last)
                line_last = arr_last.split('#')
                arr.append(line_last[0])
                arr.append(line_last[1][0])
                # print(arr)

                lines.append(arr)
        # with open(RESULTPATH, 'w', newline='') as f2:
        #     csv_write = csv.writer(f2)
        #     csv_write.writerows(lines)
        return lines
