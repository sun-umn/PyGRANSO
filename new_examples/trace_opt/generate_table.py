import numpy as np
import os
import utils
import torch
import matplotlib.pyplot as plt


my_path = os.path.dirname(os.path.abspath(__file__))

K = 200 # K number of starting points. random generated initial guesses
N = 8 # number of different data matrix (with the same size)

folding_list = ['l2','l1','linf','unfolding']
# folding_list = ['l2']

plt_data =[]

for maxfolding in folding_list:
    E_arr = np.array([])
    E_all_seeds = np.array([])
    

    sigma_arr = np.array([])
    gamma_arr = np.array([])
    T_arr = np.array([])

    code_arr = np.array([])

    for rng_seed in range(N):
        [A, U, ana_sol] = utils.data_init(rng_seed, n=10, d=5, device = torch.device('cuda'))

        data_name = 'data/'+ '04292022_10:58:30_seed_7N8K200_n10_d5_l2_l1_linf_unfolding__total200_maxtime30.npy'
        dict_name = os.path.join(my_path, data_name)
        test = np.load(dict_name, allow_pickle=True)
        
        dict_key = str(rng_seed) + maxfolding
        tmp_dict = test[()][dict_key]

        # tmp_dict = test[()]

        # print(tmp_dict['F'])
        # print(ana_sol)

        # Mean error
        # E_k = tmp_dict['E']
        E_k = np.abs(tmp_dict['F'] - ana_sol)/np.abs(ana_sol)
        E_n = np.mean(E_k) # average for each seeds

        E_arr = np.append(E_arr,E_n)
        E_all_seeds = np.append(E_all_seeds,E_k)

        sigma_n = np.var(E_k) # 1/K not 1/(K-1)
        sigma_arr = np.append(sigma_arr,sigma_n)

        ge_arr = np.abs(tmp_dict['F'] / ana_sol) >=0.99
        le_arr = np.abs(tmp_dict['F'] / ana_sol) <=1.01

        gamma_n = np.sum(np.logical_and(ge_arr,le_arr))/len(E_k)
        gamma_arr = np.append(gamma_arr,gamma_n)

        T_k = tmp_dict['time']
        T_n = np.mean(T_k)
        T_arr = np.append(T_arr,T_n)

        code = tmp_dict['term_code_pass']
        # print(code)

        code_arr = np.append(code_arr,code)

        pass

    print("folding type: {}; E {}; sigma: {}; gamma: {}; T: {}".format(maxfolding,np.mean(E_arr),np.mean(sigma_arr),np.mean(gamma_arr),np.mean(T_arr)))
    plt_data.append(E_all_seeds)
    # print(code_arr)
    code0 = np.sum(code_arr==0)/len(code_arr)*100
    code2 = np.sum(code_arr==2)/len(code_arr)*100
    code5 = np.sum(code_arr==5)/len(code_arr)*100
    code6 = np.sum(code_arr==6)/len(code_arr)*100

    print("0: {}; 2: {}; 5: {}; 6: {}".format(code0,code2,code5,code6   ))
    print(code0+code2+code5+code6)



# Adding title
plt.title("Error of different folding types")
# Creating plot
plt.boxplot(plt_data)
# Creating axes instance
plt.xticks([1, 2, 3,4], ['l2', 'l1', 'linf', 'unfolding'])
plt.show()