from abc import ABC, abstractmethod
import numpy as np

from .kernel import *

class Window(ABC):
    '''
    Simulates a window centered around a pixel (i,j) in
    an image. All it does is store some relative coords.
    You decide what to do with it.
    '''
    def __init__(self):
        pass

    @abstractmethod
    def extract_nbrs(self, shape):
        '''
        **Inputs**
        ----
        - shape [tuple]: is the shape of the image, call it (H,W)
        
        **Outputs**
        ----    
        - nbr_rows [np.ndarray]: for each (i,j) we want all N of the neighbours. This is a (N, H, W) array
        - nbr_cols [np.ndarray]: for each (i,j) we want all N of the neighbours. This is a (N, H, W) array

        **Description**
        ----
        Extracts out all the neighbours of (i,j) and returns a 3D arrays where (x,i,j) give you 
        the nighbouring pixel locations (row & cols separate). Consumes memory, but will be 
        useful for vectorization.
        '''
        pass


class square_window(Window):
    '''
    Square window. Will contain the relative indices around
    an anchor point.
    '''
    def __init__(self, n: int, kernel= None):
        '''
        **Inputs**
        ----
        - n [int]: The size of the window
        - kernel [assn2.Kernel]: a kernel of values.

        Initializes `self.coords` which is an n^2 x 2 shape
        and `self.kernel` to the required kernel flattend in row major order.
        Contains all the relative coordinate to extract pixels
        and do operations on them.
        '''
        assert isinstance(n, int) and n > 0, "n must be an integer more than zero"

        start_index= -(n//2)

        # the coordinate array will contain relative coords, array shape (N,2)
        self.coords= np.array([[i,j] for i in range(start_index, start_index+n) for j in range(start_index, start_index+n)])
        # this contains the kernel if any 
        self.kernel= box_blur_kernel(n) if kernel is None else kernel
        self.shape=(n,n)
    
    def extract_nbrs(self, shape: tuple):
        '''
        Extract out the neighbours for each pixel and give it.
        '''
        assert shape is not None, "shape cannot be None!"
        assert len(shape)== 2, "shape must be 2D otherwise indexing not possible!"

        ## 1. extract index arrays (row index and col index)
        rows, cols= np.indices(shape)

        ## 2. offset them
        nbr_rows= rows[:,:,np.newaxis] + self.coords[:, 0]
        nbr_cols= cols[:,:,np.newaxis] + self.coords[:, 1]
        # (H, W, 1) + (N,)= (H, W, N)
        # nbr_rows[i,j,:] => row indices of nbrs of (i,j)
        # nbr_cols[i,j,:] => col indices of nbrs of (i,j)

        return nbr_rows, nbr_cols