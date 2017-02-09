import sys

from load_plot import *
from gpr import *

import pandas as pd
import numpy as np

from sklearn.gaussian_process.kernels import *

kernels = []

kernels.append(Sum(RBF(length_scale=10),WhiteKernel()))
    
nus = [0.5, 1.5, 2.5]
for n in nus:
    kernels.append(Sum(Matern(length_scale=10,nu=n),WhiteKernel()))
        
kernels.append(Sum(RationalQuadratic(),WhiteKernel()))

batch_size = int(sys.argv[1])
window_size = int(sys.argv[2])
kernel = kernels[int(sys.argv[3])]

name_file = 'A4Benchmark_all.csv'
name_TS = 'A4Benchmark-TS1'
TS = load_TS(name_file,name_TS)
true,pred,mse,params = GPR(TS,batch_size,window_size,kernel,True)

np.save('mse',mse)
np.save('params',params)
np.save('pred',pred)
np.save('true',true)