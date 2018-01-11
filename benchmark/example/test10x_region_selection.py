import cProfile
import numpy as np
import time
import random
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
h5gz_gene = '/'.join([path ,"gz_chunk_by_gene.h5"])
h5gz_cell = '/'.join([path ,"gz_chunk_by_cell.h5"])

f_local = h5py.File(h5gz, "r")
ds_local = f_local["/data"]
ds_local.shape
ds_local.chunks

nGenes, nCells = ds_local.shape

f_gene = h5py.File(h5gz_gene, "r")
ds_gene = f_gene["/data"]
f_cell = h5py.File(h5gz_cell, "r")
ds_cell = f_cell["/data"]

f_remote = h5pyd.File("/home/wjiang2/10x", "r")#bug in h5pyd: mode r is still writable
ds_remote = f_remote["/data"]
ds_remote.shape
ds_remote.chunks
#===============================================================================
# benchmark - point selection
#===============================================================================

random.seed(1)

cells = genes = [10,50,100,500,1000,2000,3000,4000,5000,6000,7000]

res = np.zeros((2, len(cells)))
for i,size in enumerate(cells): 
    print(size)
    #generate random block start idx for x and y
    xstart = random.randint(0,nGenes - size)
    xend = xstart + size
    ystart = random.randint(0,nCells - size)
    yend = ystart + size
    
    #read local
    
    t1 = time.perf_counter()
    vals1 = ds_local[xstart:xend, ystart:yend]
    res[0,i] = time.perf_counter() - t1

    #hybrid approach performs worse than rect-chunked single h5
#     t1 = time.perf_counter()
#     vals2 = bm.region_selection_hybrid(ds_gene, ds_cell, (xstart, xend), (ystart, yend))
#     res[1,i] = time.perf_counter() - t1

    #read remote
    t1 = time.perf_counter()
    vals3 = ds_remote[xstart:xend, ystart:yend]
    res[1,i] = time.perf_counter() - t1
    #validity check
#     if not (np.array_equal(vals1, vals2) and np.array_equal(vals1, vals3)):
    if not np.array_equal(vals1, vals3):
        raise ValueError("read not equal!")
# cProfile.run('vals1 = bm.random_slicing_local(ds_local, idx1, idx2)', "lstats")
# cProfile.run('vals2 = bm.random_slicing_remote(ds_remote, idx1, idx2)', "rstats")
# import pstats
# p = pstats.Stats('lstats')
# p.sort_stats('cumulative').print_stats(20)

#===============================================================================
# plot the timing results
#===============================================================================
xbrks = ['$' + str(i) + '^2$' for i in cells]
df = pd.DataFrame(res.transpose(), columns = ['h5 local', 'hsds'], index = xbrks)
df.to_csv("region_sel.csv")
df = pd.read_csv("region_sel.csv", index_col = 0)
df.plot(style = '.-')
plt.yscale("log")
plt.xlabel("size of rectangular subset")
plt.ylabel("Time (s)")
plt.title("continuous region selection")
