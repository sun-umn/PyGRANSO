from sklearn import svm
# from sklearn import datasets
from sklearn.preprocessing import normalize
import numpy as np
import time


from torchvision import datasets as torch_datasets
from torchvision.transforms import ToTensor
import torch

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from sklearn.svm import LinearSVC

from sklearn import datasets

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as matplot
import os
from datetime import datetime
import sys

import bz2






max_iter = 100
write_to_log = False

########################################
# save file
now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y_%H:%M:%S")
my_path = os.path.dirname(os.path.abspath(__file__))
name_str = 'sklearnSVM_maxit{}'.format(max_iter)
png_title =  "png/" + date_time + name_str 
log_name = "log/" + date_time + name_str + ".txt"


if write_to_log:
    sys.stdout = open(os.path.join(my_path, log_name), 'w')

# make normalized binary classification dataset

# train_data = bz2.open('/home/buyun/datasets/rcv1_train.binary.bz2')

# with bz2.open('/home/buyun/datasets/rcv1_train.binary.bz2', "rb") as f:
#     # Decompress data from file
#     content = f.read()

# dfile = bz2.BZ2File('/home/buyun/datasets/rcv1_train.binary.bz2')
# mydata = np.fromfile(dfile)
# .reshape(dim,rows,cols)

X, y = datasets.load_svmlight_file('/home/buyun/datasets/rcv1_train.binary.bz2')

train_data = torch_datasets.MNIST(
    root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist',
    train = True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download = True,
)

test_data = torch_datasets.MNIST(
    root = '/home/buyun/Documents/GitHub/PyGRANSO/examples/data/mnist',
    train = False,
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download = True,
)

loaders = {
    'train' : torch.utils.data.DataLoader(train_data,
                                        batch_size=60000,
                                        shuffle=True,
                                        num_workers=1),
    'test' : torch.utils.data.DataLoader(test_data,
                                        batch_size=10000,
                                        shuffle=True,
                                        num_workers=1)
}
X_train, y_train = next(iter(loaders['train']))
X_test, y_test = next(iter(loaders['test']))
X_train = torch.reshape(X_train,(-1,28*28))
y_train[y_train%2==1] = 1
y_train[y_train%2==0] = -1
X_test = torch.reshape(X_test,(-1,28*28))
y_test[y_test%2==1] = 1
y_test[y_test%2==0] = -1
n_train = X_train.shape[0]
n_test = y_test.shape[0]
# torch to numpy
X_train = X_train.cpu().detach().numpy()
y_train = y_train.cpu().detach().numpy()
X_test = X_test.cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()

###############################################

acc = []
acc_tr = []
coefficient = []
for c in [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]:
    start = time.time()

    svm = LinearSVC(dual=False, C=c, verbose=1, max_iter=max_iter)
    svm.fit(X_train, y_train)
    coef = svm.coef_
    n_iter = svm.n_iter_
    
    p_tr = svm.predict(X_train)
    a_tr = accuracy_score(y_train, p_tr)
    
    pred = svm.predict(X_test)
    a = accuracy_score(y_test, pred)
    
    coefficient.append(coef)
    acc_tr.append(a_tr)
    acc.append(a)

    end = time.time()
    print("c = {}, n_iter = {}".format(c,n_iter))
    print('training time = {}'.format(end-start))

c = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

matplot.subplots(figsize=(10, 5))
matplot.semilogx(c, acc,'-gD' ,color='red' , label="Testing Accuracy")
matplot.semilogx(c, acc_tr,'-gD' , label="Training Accuracy")
#matplot.xticks(L,L)
matplot.grid(True)
matplot.xlabel("Cost Parameter C")
matplot.ylabel("Accuracy")
matplot.legend()
matplot.title('Accuracy versus the Cost Parameter C (log-scale)')
# matplot.show()



matplot.savefig(os.path.join(my_path, png_title))

if write_to_log:
    # end writing
    sys.stdout.close()