import re
import matplotlib.pyplot as plt
import numpy as np

alpha = 0.5

fig,ax = plt.subplots()
plt.grid(linestyle='dotted')
plt.xticks(fontsize=30)
plt.yticks(fontsize=40)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
ax.set_xscale('log')

ax2 = ax.twinx()

inf_file = "/home/buyun/Documents/GitHub/PyGRANSO/new_examples/perceptual attack/history/fig1a/log_linf.txt"

f = open(inf_file, "r")
lines = f.readlines()

obj_list = []
iteration_list = []
tv_list = []

for line in lines:

    objective = re.findall(r'\d+\s+║\s+(-*\d+[\d*.]\d+)\s+║',line)
    if objective != []:
        objective_record = float(objective[0])
        obj_list.append(objective_record)
    

    iteration = re.findall(r'^\s+(\d+)\s+║',line)
    if iteration != []:
        iteration_record = int(iteration[0])
        iteration_list.append(iteration_record)

    tv = re.findall(r'\s+(\d+.\d+[e]?[-]?\d*)\s+│   -  ║',line)
    if tv != []:
        tv_record = float(tv[0])
        tv_list.append(tv_record)

obj_list = -1*np.array(obj_list)
iteration_list = np.array(iteration_list)
tv_list = np.array(tv_list)

ax.plot(iteration_list,obj_list[0:len(iteration_list)],label='w/o R',color='#1f77b4',lw=5)
ax2.plot(iteration_list,tv_list[0:len(iteration_list)],color='#1f77b4',linestyle='dashed',lw=5,alpha=alpha)

#####################################################################################

inf_file = "/home/buyun/Documents/GitHub/PyGRANSO/new_examples/perceptual attack/history/fig1a/log_inf_reformualtion.txt"

f = open(inf_file, "r")
lines = f.readlines()

obj_list = []
iteration_list = []
tv_list = []

for line in lines:

    objective = re.findall(r'\d+\s+║\s+(-*\d+[\d*.]\d+)\s+║',line)
    if objective != []:
        objective_record = float(objective[0])
        obj_list.append(objective_record)
    

    iteration = re.findall(r'^\s+(\d+)\s+║',line)
    if iteration != []:
        iteration_record = int(iteration[0])
        iteration_list.append(iteration_record)

    tv = re.findall(r'\s+(\d+.\d+[e]?[-]?\d*)\s+│   -  ║',line)
    if tv != []:
        tv_record = float(tv[0])
        tv_list.append(tv_record)

obj_list = -1*np.array(obj_list)
iteration_list = np.array(iteration_list)
tv_list = np.array(tv_list)

# ax.plot(iteration_list,obj_list,label='w/ reformulation')
# ax2.plot(iteration_list,tv_list[0:len(iteration_list)])

ax.plot(iteration_list,obj_list[0:len(iteration_list)],label='w/ R',color='#ff7f0e',lw=5)
ax2.plot(iteration_list,tv_list[0:len(iteration_list)],color='#ff7f0e',linestyle='dashed',lw=5,alpha=alpha)

##################################################################################

inf_file = "/home/buyun/Documents/GitHub/PyGRANSO/new_examples/perceptual attack/history/fig1a/log_inf_reformualtion_folding.txt"

f = open(inf_file, "r")
lines = f.readlines()

obj_list = []
iteration_list = []
tv_list = []

for line in lines:

    objective = re.findall(r'\d+\s+║\s+(-*\d+[\d*.]\d+)\s+║',line)
    if objective != []:
        objective_record = float(objective[0])
        obj_list.append(objective_record)
    

    iteration = re.findall(r'^\s+(\d+)\s+║',line)
    if iteration != []:
        iteration_record = int(iteration[0])
        iteration_list.append(iteration_record)

    tv = re.findall(r'\s+(\d+.\d+[e]?[-]?\d*)\s+│   -  ║',line)
    if tv != []:
        tv_record = float(tv[0])
        tv_list.append(tv_record)

obj_list = -1*np.array(obj_list)
iteration_list = np.array(iteration_list)
tv_list = np.array(tv_list)

# ax.plot(iteration_list,obj_list[0:len(iteration_list)],label='w/ reformulation & folding')
# ax2.plot(iteration_list,tv_list[0:len(iteration_list)])

ax.plot(iteration_list,obj_list[0:len(iteration_list)],label='w/ R & F',color='#2ca02c',lw=5)
ax2.plot(iteration_list[10:],tv_list[10:len(iteration_list)],color='#2ca02c',linestyle='dashed',lw=5,alpha=alpha)

plt.grid(linestyle='dotted')
plt.xticks(fontsize=30)
plt.yticks(fontsize=40)
# plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)

fig.tight_layout()
plt.grid(axis="x")
plt.style.use('seaborn')
# plt.legend(loc='lower right',fontsize=10)

# ax.rc('legend', fontsize=0.5, linewidth=2)

# lgnd = ax.legend(fontsize=25,bbox_to_anchor=(0.3,0.7))
# lgnd = ax.legend(fontsize=30,loc='upper left')


plt.savefig("/home/buyun/Documents/GitHub/MinMaxGranso/log_folder/Mangi/fig1a")

plt.show()

