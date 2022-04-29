import numpy as np
import os
import utils
import torch

my_path = os.path.dirname(os.path.abspath(__file__))

K = 50 # K number of starting points. random generated initial guesses
N = 8 # number of different data matrix (with the same size)

# folding_list = ['l2','l1','linf','unfolding']
folding_list = ['unfolding']

for maxfolding in folding_list:
    E_arr = np.array([])
    sigma_arr = np.array([])
    gamma_arr = np.array([])
    T_arr = np.array([])

    for rng_seed in range(N):
        [A, U, ana_sol] = utils.data_init(rng_seed, n=10, d=5, device = torch.device('cuda'))

        data_name = 'data/'+ '04282022_18:28:08_seed_{}_n10_d5_l2_l1_linf_unfolding__total50_maxtime30.npy'.format(rng_seed)
        dict_name = os.path.join(my_path, data_name)
        test = np.load(dict_name, allow_pickle=True)
        
        # dict_key = str(rng_seed) + maxfolding
        # tmp_dict = test[()][dict_key]

        tmp_dict = test[()]

        # Mean error
        E_k = tmp_dict['E']
        E_n = np.mean(E_k) # average for each seeds

        E_arr = np.append(E_arr,E_n)

        sigma_n = np.var(E_k) # 1/K not 1/(K-1)
        sigma_arr = np.append(sigma_arr,sigma_n)

        gamma_n = np.sum(np.abs(tmp_dict['F'] - ana_sol) <= 0.01)/len(E_k)
        gamma_arr = np.append(gamma_arr,gamma_n)

        T_k = tmp_dict['time']
        T_n = np.mean(T_k)
        T_arr = np.append(T_arr,T_n)

        code = tmp_dict['term_code_pass']
        print(code)

        pass

print("E {}; sigma: {}; gamma: {}; T: {}".format(np.mean(E_arr),np.mean(sigma_arr),np.mean(gamma_arr),np.mean(T_arr)))