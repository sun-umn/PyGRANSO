import numpy as np
import os
import utils
import torch
import matplotlib.pyplot as plt


my_path = os.path.dirname(os.path.abspath(__file__))


plt_data = []

F_arr = np.array([])

sigma_arr = np.array([])
gamma_arr = np.array([])
T_arr = np.array([])

code_arr = np.array([])

tv_arr = np.array([])
opt_tol_arr = np.array([])


# data_name = 'data/'+ '05062022_10:12:13batch_size1000_maxtime300_attackPerceptual_box_cstr_seed0.npy'
data_name = 'data/'+ '05062022_20:09:55batch_size1000_maxtime300_attackPerceptual__seed0.npy'

# data_name = 'data/'+ '05072022_10:22:13batch_size1000_maxtime300_attackL_inf__seed0.npy'
dict_name = os.path.join(my_path, data_name)
test = np.load(dict_name, allow_pickle=True)

tmp_dict = test[()]


F_arr = np.append(F_arr,tmp_dict['F'])

tv_arr = np.append(tv_arr,tmp_dict['tv'])

# opt_tol_arr = np.append(opt_tol_arr,tmp_dict['tv'])




sigma = np.var(F_arr) 

# ge_arr = np.abs(tmp_dict['F'] / ana_sol) >=0.99
# le_arr = np.abs(tmp_dict['F'] / ana_sol) <=1.01

# gamma_n = np.sum(np.logical_and(ge_arr,le_arr))/len(E_k)*100
# gamma_arr = np.append(gamma_arr,gamma_n)

T_k = tmp_dict['time']
T = np.mean(T_k)

code = tmp_dict['term_code_pass']
# print(code)

code_arr = np.append(code_arr,code)

pass

print("F {:.2e}; sigma: {:.2e}; gamma: {:.2f}; T: {:.2f}".format(np.mean(F_arr),np.mean(sigma_arr),np.mean(gamma_arr),np.mean(T_arr)))
plt_data.append(F_arr)
# print(code_arr)
code0 = np.sum(code_arr==0)/len(code_arr)*100
code2 = np.sum(code_arr==2)/len(code_arr)*100
code5 = np.sum(code_arr==5)/len(code_arr)*100
code6 = np.sum(code_arr==6)/len(code_arr)*100

print("0: {:.2f}; 2: {:.2f}; 5: {:.2f}; 6: {:.2f}".format(code0,code2,code5,code6   ))
print(code0+code2+code5+code6)

print('tv mean = {:.2e}; tv var = {:.2e}'.format(np.mean(tv_arr),np.var(tv_arr)))
print('opt_tol mean = {:.2e}; opt_tol var = {:.2e}'.format(np.mean(opt_tol_arr),np.var(opt_tol_arr)))

# # Adding title
# plt.title("Error of different folding types")
# # Creating plot
# plt.boxplot(plt_data)
# # Creating axes instance
# # plt.xticks([1, 2, 3,4], ['l2', 'l1', 'linf', 'unfolding'])
# plt.xticks([1, 2, 3], ['l2_sq', 'l1_sq', 'linf_sq'])
# plt.ylim([-0.05, 0.85])
# plt.locator_params(axis="y", nbins=5)
# plt.ylabel('Error')

# plt.show()