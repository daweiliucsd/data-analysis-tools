"""
used to gather all DA results from different files and generate a table
written by Dawei Li, March 2019
"""

import os
import numpy as np
import pandas as pd

# set the names of columns in param_values list, need to be modified depending on how you plan to organize this table, please search the .split function before using
def load_params(id_num, dirs, path):
    params = np.loadtxt('{0}/params{1}.dat'.format(dirs, path), dtype = str)
    param_values = [str(id_num), dirs.split(os.sep)[1], dirs.split(os.sep)[2], dirs.split(os.sep)[3].split('_')[1]] + list(params[:,-1].astype(str))
    return np.array(param_values)

# Here is the number of initial conditions you set after -t in sub.sh file
num_paths = 5

# # used to select results that have cost funtion from a certain range
# upper_bound = 1000
# lower_bound = 0

labels = []
for root, dirs, files in os.walk(".", topdown=False):
    if os.path.exists('{0}/i.dat'.format(root)):
        labels.append(root)

for path in range(1, num_paths+1):
    try:
        column_names = ['path_id', 'bird', 'neuron', 'stim'] + list(np.loadtxt('{0}/params{1}.dat'.format(labels[0], path), dtype = str)[:,0]) + ['cost_value']
        break
    except:
        continue

rows = np.array([], dtype = np.str).reshape(0, len(column_names))

for dirs in labels:
    for path in range(1, num_paths+1):
        id_num = path
        try:
            Obj_value = np.loadtxt('{0}/Obj{1}.dat'.format(dirs, path), delimiter = ',')
            #if float(Obj_value) > upper_bound or float(Obj_value) < lower_bound:
                #continue
        except:
            continue
        new_row = np.append(load_params(id_num, dirs, path), Obj_value)
        rows = np.vstack((rows, new_row))

df = pd.DataFrame(rows, columns = column_names)

df.to_csv('param_table.csv')
