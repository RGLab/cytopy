import h5py
import h5pyd
import numpy as np
from random import sample
from random import seed
import time
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
from h5pyd._apps.hsdel import deleteDomain 

#===============================================================================
# access local H5
#===============================================================================
f_local = h5py.File("/home/wjiang2/rglab/workspace/cytopy/benchmark/data/test.h5", "r")
ds_local = f_local["/28"]
ds_local.shape
ds_local.chunks
nChannel, nCells = ds_local.shape
f_local.libver
def read_local(ds_local, idx1, idx2, isPointSelection = True):
    """random slicing local H5 (2d)
        ds_local h5py._hl.dataset.Dataset
        idx1 sorted random idx for dimension 1
        idx2 sorted random idx for dimension 2
        isPointSelection boolean whether do the point selection at H5 level
    """
    #preallocate output memory 
    vals1 = np.zeros((len(idx1), len(idx2)), dtype='float32')
    t1 = time.perf_counter()
    #fix x dim and random slicing on y since h5py doesn't allow random selection on both 
    for i,x in enumerate(idx1):
        if(isPointSelection):
            vals1[i,:] = ds_local[x, idx2]
        else:
            vals1[i,:] = ds_local[x, :][idx2]
    t2 = time.perf_counter() - t1
    return t2


#===============================================================================
# create remote h5
#===============================================================================
# deleteDomain("/home/wjiang2/tcell") 
# f_remote = h5pyd.File("/home/wjiang2/tcell", "w")
# # f_remote.create_group("datasets")
#  
# f_remote['/0'].chunks
# d_remote = f_remote.create_dataset("0", (nChannel, nCells), "float32", chunks = (5,120727))
# for i in range(nChannel):
#     d_remote[i, :] = ds_local[i, :] 
# f_remote.close()
     

#===============================================================================
# API to access remote h5 
#===============================================================================

def read_remote(ds_remote, idx1, idx2):
    """random slicing local H5 (2d)
        ds_remote h5py._hl.dataset.Dataset
        idx1 sorted random idx for dimension 1
        idx2 sorted random idx for dimension 2
    """
    
    #use tupled coordinates to do point selection directly through h5pyd
    coord = [(i,j) for i in idx1 for j in idx2]
    t = time.perf_counter()
    vals2 = ds_remote[coord]
    t2 = time.perf_counter() - t
    return t2

#h5pyd returns the flattened 1d array
#fortunately there is cheap way to reshape it
# vals2.shape = (nX, nY)
# np.array_equal(vals1, vals2)

f_remote = h5pyd.File("/home/wjiang2/tcell", "r")
ds_remote = f_remote["/0"]
ds_remote.shape
ds_remote.chunks
#===============================================================================
# benchmark
#===============================================================================

seed(1)

nX = 2 #fix the number of channels
cells = list(range(10, 10000,1000))#different size of cells

res = np.zeros((3, len(cells)))
for i,nY in enumerate(cells): 
    #generate random idx for x and y
    idx1 = sorted(sample(range(nChannel), nX))
    idx2 = sorted(sample(range(nCells), nY))
    res[0,i] = read_local(ds_local, idx1, idx2, isPointSelection = False)
    res[1,i] = read_local(ds_local, idx1, idx2, isPointSelection = True)
    res[2,i] = read_remote(ds_remote,idx1, idx2)

#===============================================================================
# plot the timing results
#===============================================================================
df = pd.DataFrame(res.transpose(), columns = ['h5 local(non-pt-sel)', 'h5 local' ,'hsds'], index = cells)
df.plot(style = '.-')
plt.yscale("log")