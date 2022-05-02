import utils
import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso

maxfolding = 'l2'
maxclocktime = 600
maxit = 1200

device = torch.device('cuda')

sequence_length = 28*28
input_size = 1

# sequence_length = 28
# input_size = 28

hidden_size = 30
# hidden_size = 200

num_layers = 1
num_classes = 10

train_size=100 # train acc 100; test acc 19%
test_size=100

double_precision = torch.double
# double_precision = torch.float

rng_seed = 5
# rng_seed = 6

###############################################
model = utils.get_model(rng_seed,input_size, hidden_size, num_layers, num_classes,device,double_precision,sequence_length)

[X_train, y_train, X_test, y_test] = utils.get_data(sequence_length, input_size, device, double_precision,train_size,test_size)

# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

comb_fn = lambda model : utils.user_fn(model,X_train,y_train,hidden_size,device,double_precision,maxfolding)

opts = utils.get_opts(device,model,maxit,maxclocktime,double_precision)


utils.get_model_acc(model,X_train,y_train,train=True)
utils.get_model_acc(model,X_test,y_test,train=False)

start = time.time()
soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))

torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())

utils.get_model_acc(model,X_train,y_train,train=True)
utils.get_model_acc(model,X_test,y_test,train=False)
