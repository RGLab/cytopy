#import h5py
#import h5pyd
import numpy as np

def point_select_local(ds, coord, data):
    """random point selection from local H5 (2d) by point coordinates
        Args:
            ds (h5py._hl.dataset.Dataset): source dataset object
            coord (tuple): the coordinates of points to read
        Returns:
            numpy.ndarray:  1d array
    """
    data = np.zeros((len(coord)), dtype='float32')
    for i,item in enumerate(coord):
        data[i] = ds[item]
    return data

def random_slicing_local(ds, ridx, cidx):
    """slicing from local H5 by row idx and col idx
        Args:
            ds (h5py._hl.dataset.Dataset): source dataset object
            ridx (list): the indices of selected rows (sorted)
            cidx (list): the indices of selected columns (sorted)
        Returns:
            numpy.ndarray: 2d array
        
    """
    data = np.zeros((len(ridx),len(cidx)), dtype='float32')
    #fix x dim and random slicing on y since h5py doesn't allow random selection on both 
    for j,x in enumerate(cidx):
            data[:,j] = ds[ridx, x]
    return data
     
def random_slicing_remote(ds, ridx, cidx, fast = True):
    """slicing from hsds by row idx and col idx
       Args:
           ds (h5pyd._hl.dataset.Dataset): source dataset object
           ridx (list): the indices of selected rows (sorted)
           cidx (list): the indices of selected columns (sorted)
      Returns:
            numpy.ndarray: 2d array
       
    """
    #use point selection since h5pyd doesn't support fancy slicing yet 
    nrow = len(ridx)
    ncol = len(cidx)
    if fast:#construct global coords and submit single http request to reduce overhead
        coords = [(i, j) for i in ridx for j in cidx]
        data = ds[coords]
        data = data.reshape((nrow, ncol))
    else:
        data = np.zeros((nrow,ncol), dtype='float32')
        for j,x in enumerate(cidx):#fix cidx and read one column at a time
            coord = [(i, x) for i in ridx]
            data[:,j] = ds[coord]
    return data
