from abc import ABC, abstractmethod
import numpy as np

class Kernel(ABC):
    '''
    Implements the abstract kernel.
    '''
    def __init__(self):
        pass

class box_blur_kernel(Kernel):
    '''
    implements the box blur Kernel of a particular size
    '''
    def __init__(self, n):
        # sum of the elements in the kernel should be 1
        self.patch= np.full((n**2,), n**-2)
        self.size= self.patch.size
        self.shape= (n,n)
        
class gaussian_kernel(Kernel):
    '''
    implements the Gaussian Kernel
    '''
    def __init__(self, n, var):
        # 1. Create a grid of (y, x) coordinates relative to the center
        # idx[0] is y-coords, idx[1] is x-coords
        idx = np.indices((n, n)) - n // 2
        
        # 2. Calculate the squared Euclidean distance from the center
        # This correctly calculates (x² + y²) for each point in the grid
        squared_distance = np.sum(idx**2, axis=0)
        
        # 3. Apply the correct 2D Gaussian formula
        # Note the denominator is (2 * var), not sqrt(...)
        kernel_2d = np.exp(-squared_distance / (2 * var)) / (2 * np.pi * var)
        
        # 4. CRITICAL STEP: Normalize the kernel to sum to 1
        # This prevents the filter from changing the image brightness
        normalized_kernel = kernel_2d / np.sum(kernel_2d)
        
        # 5. Store the flattened kernel and other properties
        self.patch = normalized_kernel.flatten()
        self.var = var
        self.size = self.patch.size
        self.shape = (n, n)

class laplacian_kernel(Kernel):
    '''
    implements the full laplacian kernel
    4nbr 3x3 one is separate
    '''
    def __init__(self, n: int):
        assert n%2 == 1, "Not implemented for even kernel size"

        self.patch= np.full((n**2, ), -1)
        self.patch[n**2//2]= n**2-1
        self.size = self.patch.size
        self.shape = (n, n)
