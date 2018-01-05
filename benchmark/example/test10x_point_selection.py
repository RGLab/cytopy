import cProfile
import numpy as np
import time
from random import sample
from random import seed
import h5py
import h5pyd
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
# from h5pyd._apps.hsdel import deleteDomain 
import benchmark.h5IO as bm
import importlib
dir(bm)
importlib.reload(bm)

path = "/loc/no-backup/mike/shared/1M_neurons"
h5gz = '/'.join([path ,"gz.h5"])
f_local = h5py.File(h5gz, "r")
ds_local = f_local["/data"]
ds_local.shape
ds_local.chunks

nGenes, nCells = ds_local.shape

f_remote = h5pyd.File("/home/wjiang2/10x", "r")#bug in h5pyd: mode r is still writable
ds_remote = f_remote["/data"]
ds_remote.shape
ds_remote.chunks
#===============================================================================
# benchmark - point selection
#===============================================================================

seed(1)

cells = genes = [10,50,100,500,1000,5000,10000][0:2]

res = np.zeros((2, len(cells)))
for i,size in enumerate(cells): 
    print(size)
    #generate random idx for x and y
    idx1 = sorted(sample(range(nGenes), size))
    idx2 = sorted(sample(range(nCells), size))
    coord = list(zip(idx1, idx2))
    #read local
    
    t1 = time.perf_counter()
    vals1 = bm.point_select_local(ds_local, coord)
    res[0,i] = time.perf_counter() - t1
    #read remote
    t1 = time.perf_counter()
    vals2 = ds_remote[coord]
    res[1,i] = time.perf_counter() - t1
    #validity check
    if not np.array_equal(vals1, vals2):
        raise ValueError("read not equal!")

#===============================================================================
# plot the timing results
#===============================================================================
df = pd.DataFrame(res.transpose(), columns = ['h5 local' ,'hsds'], index = cells)
df.to_csv("pt_sel.csv")
df.plot(style = '.-')
plt.yscale("log")
plt.xlabel("# of points selected")
plt.ylabel("Time (s)")
plt.title("Random point selection")
