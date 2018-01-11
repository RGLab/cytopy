import h5py
# import h5pyd
import numpy as np
import random
import string
from random import sample
from random import seed
import time
import pandas as pd
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
# from h5pyd._apps.hsdel import deleteDomain 

path = "/loc/no-backup/mike/shared/jnj"
#===============================================================================
# rename the ds with long names
#===============================================================================
# f_old = h5py.File(path + "/old_ver_longname.h5", "r+")
# with open(path + "/longnames.txt", "r") as f_name:
#     longnames = f_name.read()
#     longnames = longnames.splitlines()
# 
# for i,name in enumerate(longnames):
#     f_old[name] = f_old[str(i)]
#     del f_old[str(i)]
# list(f_old.keys())
# f_old.close()

#===============================================================================
# simulate data with different format
#===============================================================================
# f_old_short = h5py.File(path + "/old_ver_shortname.h5", "w")
# f_old_long = h5py.File(path + "/old_ver_longname.h5", "w")
# f_new_long = h5py.File(path + "/new_ver_longname.h5", "w", libver = "latest")
# 
# rng = range(1000)
# shortnames = [str(i) for i in rng] 
# longnames = [''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)]) for i in rng]
# shape = (16, 100000)
# for i in rng:
#     f_old_short.create_dataset(shortnames[i], shape = shape, dtype="float32")
#     f_old_long.create_dataset(longnames[i],  shape = shape,dtype="float32")
#     f_new_long.create_dataset(longnames[i],  shape = shape,dtype="float32")
#     
# f_old_short.close()
# f_old_long.close()
# f_new_long.close()

#===============================================================================
# timing the ds lookup
#===============================================================================
f_old_short = h5py.File(path + "/old_ver_shortname.h5", "r")
f_old_long = h5py.File(path + "/old_ver_longname.h5", "r")
f_new_long = h5py.File(path + "/new_ver_longname.h5", "r")


def ds_lookup(f, names):
    """random link lookup from local h5
        Args:
            f (h5py._hl.files.file)
            names (list) dataset names to lookup
        Return:
            average time
    """
    #preallocate output memory 
    t1 = time.perf_counter()
    #fix x dim and random slicing on y since h5py doesn't allow random selection on both 
    a = [f[i].name for i in names]
    t2 = time.perf_counter() - t1
    a = [i.lstrip("/") for i in a]
    if a != names:
        raise ValueError("failed")
    return t2/len(names)



#===============================================================================
# benchmark --lookup
#===============================================================================
short_names = list(f_old_short.keys())
long_names = list(f_old_long.keys())
nSamples = len(short_names)
nSize = 100 #fix the number of channels

seed(1)

res = []
name_length = ["short", "long", "long"]
libver = ["earliest", "earliest", "latest"]
#generate random idx for x and y

idx1 = sample(short_names, nSize)
idx2 = sample(long_names, nSize)
res.append(ds_lookup(f_old_short, idx1))
res.append(ds_lookup(f_old_long, idx2))
res.append(ds_lookup(f_new_long, idx2))
#===============================================================================
# plot the timing results
#===============================================================================
import seaborn as sns


#===============================================================================
# benchmark --lookup
#===============================================================================

import benchmark.h5IO as bm
ds1 = f_old_short[short_names[1]]
ds2 = f_old_long[long_names[1]]
ds3 = f_new_long[long_names[1]]
ds1.shape
nX = 2
nY = 1000
seed(1)
idx1 = sorted(sample(range(16), nX))
idx2 = sorted(sample(range(10000), nY))
coord = [(i,j) for i in idx1 for j in idx2]
#read local
res1 = []
t1 = time.perf_counter()
vals1 = bm.point_select_local(ds1, coord)
res1.append(time.perf_counter() - t1)
t1 = time.perf_counter()
vals1 = bm.point_select_local(ds2, coord)
res1.append(time.perf_counter() - t1)
t1 = time.perf_counter()
vals1 = bm.point_select_local(ds3, coord)
res1.append(time.perf_counter() - t1)
res1

res
df = pd.DataFrame({'name length':name_length, 'libver':libver, 'dataset lookup':res, 'data point selection':res1})
df = df[1:3]
# df = pd.melt(df, id_vars=['libver'], value_vars=['dataset lookup','data point selection'])
g = sns.factorplot(x="variable", y="value", hue="libver", data=df, kind="bar")
