import numpy as np
import os

my_path = os.path.dirname(os.path.abspath(__file__))
data_name = 'data/'+ '04282022_18:11:21_seed_1_n10_d5_l2_unfolding__total3_maxtime30.npy'
dict_name = os.path.join(my_path, data_name)


test = np.load(dict_name, allow_pickle=True)
time = test[()]['time']
pass
