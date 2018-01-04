import numpy as np
import time
from random import sample
from random import seed
import h5py
import h5pyd
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
from h5pyd._apps.hsdel import deleteDomain 
from benchmark.h5IO import point_select_local
import importlib
dir(benchmark)
importlib.reload(benchmark)
#===============================================================================
# access local H5
#===============================================================================
path = "/loc/no-backup/mike/shared/1M_neurons"
# tenxFile = '/'.join([path ,"1M_neurons_filtered_gene_bc_matrices_h5.h5"])

h5gz = '/'.join([path ,"gz.h5"])
h5gz_gene = '/'.join([path ,"gz_chunk_by_gene.h5"])
h5gz_cell = '/'.join([path ,"gz_chunk_by_cell.h5"])

#open ds for local h5
f_local = h5py.File(h5gz, "r")
ds_local = f_local["/data"]
ds_local.shape
ds_local.chunks

nGenes, nCells = ds_local.shape

#open ds for local hybrid
f_gene = h5py.File(h5gz_gene, "r")
ds_gene = f_gene["/data"]
ds_gene.chunks
f_cell = h5py.File(h5gz_cell, "r")
ds_cell = f_cell["/data"]
ds_cell.chunks

f_remote = h5pyd.File("/home/wjiang2/10x", "r")#bug in h5pyd: mode r is still writable
ds_remote = f_remote["/data"]
ds_remote.shape
ds_remote.chunks
#===============================================================================
# benchmark - point selection
#===============================================================================

seed(1)

genes = list(range(10, 10000,1000))
cells = list(range(10, 10000,1000))#different size of cells

res = np.zeros((2, len(cells)))
for i,nSize in enumerate(cells): 
    #generate random idx for x and y
    idx1 = sorted(sample(range(nGenes), nSize))
    idx2 = sorted(sample(range(nCells), nSize))
    coord = list(zip(idx1, idx2))
    #read local
    vals1 = np.zeros((nSize), dtype='float32')
    t1 = time.perf_counter()
    point_select_local(ds_local, coord, vals1)
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
df.plot(style = '.-')
plt.yscale("log")
plt.xlabel("Number of points randomly selected")
plt.ylabel("Time (s)")
plt.title("Random point selection")
