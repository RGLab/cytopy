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
path = "/loc/no-backup/mike/shared/1M_neurons"
# tenxFile = '/'.join([path ,"1M_neurons_filtered_gene_bc_matrices_h5.h5"])

h5gz = '/'.join([path ,"gz.h5"])
h5gz_gene = '/'.join([path ,"gz_chunk_by_gene.h5"])
# h5gz_cell = '/'.join([path ,"gz_chunk_by_cell.h5"])
# h5lz_gene = '/'.join([path ,"lz_chunk_by_gene.h5"])
# h5lz_cell = '/'.join([path ,"lz_chunk_by_cell.h5"])

#===============================================================================
# create local h5
#===============================================================================

# f_local1 = h5py.File(h5gz, "w")
# f_local1['/data'].chunks
# ds = f_local1.create_dataset("data", (nGenes, nCells), "float32", chunks = (218, 20408),compression = "gzip")
# for i in range(nGenes):
#     ds[i, :] = ds_local[i, :] 
# f_local1.close()

# f = open("/loc/no-backup/mike/shared/1M_neurons/log.txt", "w")
# f.write("done")
# f.close()



f_local = h5py.File(h5gz_gene, "r")
# f_local = h5py.File(h5gz, "r")
ds_local = f_local["/data"]
ds_local.shape
ds_local.chunks
nGenes, nCells = ds_local.shape

def read_local(ds_local, idx1, idx2):
    """random slicing local H5 (2d)
        ds_local h5py._hl.dataset.Dataset
        idx1 sorted random idx for dimension 1
        idx2 sorted random idx for dimension 2
        isPointSelection boolean whether do the point selection at H5 level
    """
    coord = list(zip(idx1, idx2))
    #preallocate output memory 
    vals1 = np.zeros((len(coord)), dtype='float32')
    t1 = time.perf_counter()
    #fix x dim and random slicing on y since h5py doesn't allow random selection on both 
    for i,item in enumerate(coord):
        vals1[i] = ds_local[item]
    t2 = time.perf_counter() - t1
    return t2


#===============================================================================
# create remote h5
#===============================================================================
deleteDomain("/home/wjiang2/10x") 
f_remote = h5pyd.File("/home/wjiang2/10x", "w")
# f_remote.create_group("datasets")
  
f_remote['/data'].chunks
d_remote = f_remote.create_dataset("data", (nGenes, nCells), "float32")
for i in range(nGenes):
    d_remote[i, :] = ds_local[i, :] 
f_remote.close()
     

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
    coord = list(zip(idx1, idx2))
    t = time.perf_counter()
    vals2 = ds_remote[coord]
    t2 = time.perf_counter() - t
    return t2

#h5pyd returns the flattened 1d array
#fortunately there is cheap way to reshape it
# vals2.shape = (nX, nY)
# np.array_equal(vals1, vals2)

f_remote = h5pyd.File("/home/wjiang2/10x", "r")
ds_remote = f_remote["/data"]
ds_remote.shape
ds_remote.chunks
#===============================================================================
# benchmark
#===============================================================================

seed(1)

genes = list(range(10, 10000,1000))
cells = list(range(10, 10000,1000))#different size of cells

res = np.zeros((2, len(cells)))
for i,nSize in enumerate(cells): 
    #generate random idx for x and y
    idx1 = sorted(sample(range(nGenes), nSize))
    idx2 = sorted(sample(range(nCells), nSize))
    res[0,i] = read_local(ds_local, idx1, idx2)
#     res[1,i] = read_local(ds_local, idx1, idx2, isPointSelection = True)
    res[1,i] = read_remote(ds_remote,idx1, idx2)

#===============================================================================
# plot the timing results
#===============================================================================
df = pd.DataFrame(res.transpose(), columns = ['h5 local' ,'hsds'], index = cells)
df.plot(style = '.-')
plt.yscale("log")
plt.xlabel("Number of points randomly selected")
plt.ylabel("Time (s)")

