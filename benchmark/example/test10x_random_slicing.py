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
#===============================================================================
# access local H5
#===============================================================================
path = "/loc/no-backup/mike/shared/1M_neurons"
# tenxFile = '/'.join([path ,"1M_neurons_filtered_gene_bc_matrices_h5.h5"])

h5gz = '/'.join([path ,"gz.h5"])
# h5gz_gene = '/'.join([path ,"gz_chunk_by_gene.h5"])
# h5gz_cell = '/'.join([path ,"gz_chunk_by_cell.h5"])

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


f_local = h5py.File(h5gz, "r")
ds_local = f_local["/data"]
ds_local.shape
ds_local.chunks

nGenes, nCells = ds_local.shape

#===============================================================================
# create remote h5
#===============================================================================
# deleteDomain("/home/wjiang2/10x") 
# f_remote = h5pyd.File("/home/wjiang2/10x", "w")
# # f_remote.create_group("datasets")
#   
# f_remote['/data'].chunks
# ds_remote = f_remote.create_dataset("data", (nGenes, nCells), "float32")
# t1 = time.perf_counter()
# for i in range(nGenes):
#     ds_remote[i, :] = ds_local[i, :] 
# t2 = time.perf_counter() - t1
# f_remote.close()
     


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
    #read local
    t1 = time.perf_counter()
    vals1 = bm.random_slicing_local(ds_local, idx1, idx2)
    res[0,i] = time.perf_counter() - t1
    #read remote
    t1 = time.perf_counter()
    vals2 = bm.random_slicing_remote(ds_remote, idx1, idx2)
    res[1,i] = time.perf_counter() - t1
    #validity check
    if not np.array_equal(vals1, vals2):
        raise ValueError("read not equal!")
# cProfile.run('vals1 = bm.random_slicing_local(ds_local, idx1, idx2)', "lstats")
# cProfile.run('vals2 = bm.random_slicing_remote(ds_remote, idx1, idx2)', "rstats")
# import pstats
# p = pstats.Stats('lstats')
# p.sort_stats('cumulative').print_stats(20)

#===============================================================================
# plot the timing results
#===============================================================================
df = pd.DataFrame(res.transpose(), columns = ['h5 local' ,'hsds'], index = [(str(i) + '*' + str(i)) for i in cells])
df.to_csv("pt_sel.csv")
df.plot(style = '.-')
plt.yscale("log")
plt.xlabel("size of rectangular subset")
plt.ylabel("Time (s)")
plt.title("Random point selection")
