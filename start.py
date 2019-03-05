"""
used to put all data files to separate current(i.dat) and voltage(v.dat) files automatically, then plot all the V_t, I_t curves
written by Dawei Li, March 2019
"""

import os
import numpy as np
from subprocess import call
import matplotlib.pyplot as plt

owd = os.getcwd()
labels = []
all_files = []
for root, dirs, files in os.walk(owd, topdown=False):
	os.chdir(root)
	for file_name in files:
        
        # get rid of all the files other than data file
		if 'Anna' not in file_name:
			#print('not a Episode file', root, file_name)
			continue
	
		file_name = file_name.split('.')[0]
		call('mkdir {0}'.format(file_name), shell = True)
	
		try:
			text = np.loadtxt(file_name + '.txt', dtype=float, comments='#', delimiter='\t')
		except:
			print('loadtxt Error', root, file_name)
			continue
	
		voltage = text[:,1]
		current = text[:,0]
		time = 0.02*np.arange(len(current))
		np.savetxt('{0}/v.dat'.format(file_name), voltage)
		np.savetxt('{0}/i.dat'.format(file_name), current)
		
		plt.figure(figsize=(20,10))
		plt.subplot(2,1,1)
		plt.plot(time, voltage, color = 'black', label="Voltage", linewidth = 2)
		
		plt.subplot(2,1,2)
		plt.plot(time, current, color = 'purple', label = "Injected Current", linewidth = 2)
		
		plt.xlabel("time (ms)", fontsize = 20)
		plt.ylabel("current (pA)", fontsize = 20)
		plt.legend()
		#plt.savefig('{0}/V_I.png'.format(file_name))
		plt.savefig('{0}.png'.format(file_name))
		
		plt.close()
