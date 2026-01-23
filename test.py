import gridutils
import numpy as np
from mpi4py import MPI

def model(x):
    x1, x2, x3, x4 = x
    result = {'sum': np.array([x1 + x2 + x3 + x4])}
    return result

def get_gridvals():
    x1 = np.arange(1,10.01,1)
    x2 = np.arange(1,10.01,1)
    x3 = np.arange(1,10.01,1)
    x4 = np.arange(1,10.01,1)
    gridvals = (x1, x2, x3, x4)
    return gridvals

if __name__ == '__main__':

    gridutils.make_grid(
        model_func=model,
        gridvals=get_gridvals(),
        filename='test.h5',
        progress_filename='test.log'

    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"rank{rank}")