"""
used to submit all data files at the same time to BOOM cluster automatically
written by Dawei Li, March 2019
"""

from subprocess import call
import os
import shutil

# here are two functions that can used to modify any lines in DA configuration files, such as specs.txt, {problem_name}.opt, sub.sh

# used to add lines in a text file
def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
        f.close()

# used to replace certain line in a text file
def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

owd = os.getcwd()

#f=open('sub.sh','a')
#f.write('{0}/biohh1_cpp $SGE_TASK_ID'.format(owd))
#f.close()
#replace_line('sub.sh', 12, '{0}/biohh1_cpp $SGE_TASK_ID'.format(owd))

# search paths of all files in current directory that have a input file then save it as a list
labels = []
for root, dirs, files in os.walk('.', topdown=False):
    if os.path.exists('{0}/i.dat'.format(root)):
        labels.append(root)


for dirs in labels:
    # copy required DA files to those paths
    shutil.copy('specs.txt', dirs)
    shutil.copy('sub.sh', dirs)
    shutil.copy('biohh1.opt', dirs)
    
    ## you can change the DA configuration inside "specs.txt" here
    # line_prepender('{0}/specs.txt'.format(dirs),\
    #                '18000\n'+'105000\n'+'0.04\n'+'./Vs_noisy_50kHz.dat\n'+'./current_50kHz.dat')

    # submit jobs in all paths automatically
    os.chdir(dirs)
    call('qsub sub.sh', shell=True)
    os.chdir(owd)
