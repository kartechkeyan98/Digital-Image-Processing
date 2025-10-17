from abc import ABC, abstractmethod
import numpy as np

from .constants import *
from .window import square_window
from .kernel import box_blur_kernel, gaussian_kernel, laplacian_kernel

class Filter(ABC):
    '''
    Implements an abstract filter class, which actual
    working filters inherit from.
    '''

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        '''
        **Inputs**
        ----
        - image [np.ndarray]: image which is to be operated on.
        
        **Outputs**
        ----    
        - final_img [np.ndarray]: final image after operating on it

        **Description**
        ----
        Takes in a image and outputs another image
        '''
        pass
    
    def __call__(self, image: np.ndarray, *args, **kwargs):
        '''
        Just a way to directly call the apply method instead of 
        '''
        return self.apply(image, *args, **kwargs)

# ------------------------------------------------------------------ #
# ------------- 1. Kernel Based Blur Filters ----------------------- #
# ------------------------------------------------------------------ #

# ------------- 1.1 Box Blurring Filter ---------------------------- #
class box_filter(Filter):
    '''
    The goal of this filter is to take a binary image and 
    given a window, will do box blur on the image
    '''
    def __init__(self, n: int):
        '''
        **Inputs**
        ----
        - n [int]: sqaure window size (>= 1)
        
        **Outputs**
        ----    
        None

        **Description**
        ----
        creates a window with box blur kernel and a set of relative coordinates given n.
        The handling of odd and even windows is different.
        '''
        self.window= square_window(n, kernel= box_blur_kernel(n))
    
    def apply(self, image, mode='constant', const_value= 0):
        '''
        **Inputs**
        ----
        - image [np.ndarray]: image which is to be operated on, (H,W) or (H,W,C) is fine as well
        - mode [str]: padding mode, whether replicate padding or constant value padding.
        - const_value [float | np.ndarray]
        
        **Outputs**
        ----    
        - final_img [np.ndarray]: final image after operating on it of same dims as input image

        **Description**
        ----
        Takes in an image (can be rgb, grayscale or binary) and outputs the box blurred image
        '''
        assert image is not None, "Image matrix cannot be \33[1mNone\33[0m!"
        assert isinstance(image,(np.ndarray,)) and len(image.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
        assert isinstance(const_value, (float, int, np.ndarray)), "\033[1mconst_value\33[0m must be a numerical type or a numpy.ndarray with dim. 1"

        ## 1. extract the neighboring rows and cols
        nbr_rows, nbr_cols= self.window.extract_nbrs(image.shape[:2])
        # shapes of these are both (H,W,N)
        H, W= image.shape[:2]
        N= self.window.kernel.patch.size

        ## 2. in case replicate padding used, clip the out of bounds to inside range
        if mode=='replicate':
            np.clip(nbr_rows, 0, H-1, out=nbr_rows)
            np.clip(nbr_cols, 0, W-1, out=nbr_cols)
        
            # this should be of shape (H,W,N,C) if image has C channels
            # this should be of shape (H,W,N) if image is grayscale
            gathered_nbrs= image[nbr_rows, nbr_cols]

        ## 3. In case constant i.e. fixed padding used, then make the out of bounds 
        # pixels of constant value given above
        if mode=='constant':
            out_of_bounds= (nbr_rows < 0) | (nbr_rows >= H) | \
                        (nbr_cols < 0) | (nbr_cols >= W)
            np.clip(nbr_rows, 0, H-1, out=nbr_rows)
            np.clip(nbr_cols, 0, W-1, out=nbr_cols)
        
            # this should be of shape (H,W,N,C) if image has C channels
            # this should be of shape (H,W,N) if image is grayscale
            gathered_nbrs= image[nbr_rows, nbr_cols]
            gathered_nbrs[out_of_bounds]= const_value

            # these are done for getting back memory
            del out_of_bounds
        
        # these are done for getting back memory since they can be very large objects
        del nbr_cols
        del nbr_rows
        
        ## 4. Get the kernel, shape should be (N,) or (N,C)
        ker= self.window.kernel.patch

        mod_image= gathered_nbrs.reshape((H,W,N,-1)) * ker.reshape((1,1,N,-1))
        
        ## 5. mod_image has shape (H,W,N,C) and contains the convolution
        return np.sum(mod_image, axis=2)


# ------------- 1.2 Box Blurring Filter ---------------------------- #
class gaussian_filter(Filter):
    '''
    The goal of this filter is to take a binary image and 
    given a window, will do Gaussian blur on the image
    '''
    def __init__(self, n: int, var: float):
        '''
        **Inputs**
        ----
        - n [int]: sqaure window size (>= 1)
        - var [float]: variance of the Gaussian kernel
        
        **Outputs**
        ----    
        None

        **Description**
        ----
        creates a window ie a set of relative coordinates given n.
        The handling of odd and even windows is different.
        '''
        self.window= square_window(n, kernel= gaussian_kernel(n, var))
    
    def apply(self, image, mode='constant', const_value= 0):
        '''
        **Inputs**
        ----
        - image [np.ndarray]: image which is to be operated on, (H,W) or (H,W,C) is fine as well
        - mode [str]: padding mode, whether replicate padding or constant value padding.
        - const_value [float | np.ndarray]
        
        **Outputs**
        ----    
        - final_img [np.ndarray]: final image after operating on it of same dims as input image

        **Description**
        ----
        Takes in an image (can be rgb, grayscale or binary) and outputs the Gaussian blurred image
        '''
        assert image is not None, "Image matrix cannot be \33[1mNone\33[0m!"
        assert isinstance(image,(np.ndarray,)) and len(image.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
        assert isinstance(const_value, (float, int, np.ndarray)), "\033[1mconst_value\33[0m must be a numerical type or a numpy.ndarray with dim. 1"

        # 1. --- Extract Neighbor Coordinates ---
        H, W = image.shape[:2]
        nbr_rows, nbr_cols = self.window.extract_nbrs((H, W)) # Shape: (H, W, N)

        # 2. --- Gather Pixels with Boundary Handling ---
        # The common logic is now outside the 'if' statements.
        
        # First, identify out-of-bounds locations (only needed for 'constant' mode)
        if mode == 'constant':
            out_of_bounds = (nbr_rows < 0) | (nbr_rows >= H) | \
                            (nbr_cols < 0) | (nbr_cols >= W)

        # Clip indices to prevent errors. This is safe for both modes.
        np.clip(nbr_rows, 0, H - 1, out=nbr_rows)
        np.clip(nbr_cols, 0, W - 1, out=nbr_cols)
        
        # Gather all neighbors in one go. Works for both grayscale and color.
        gathered_nbrs = image[nbr_rows, nbr_cols] # Shape: (H, W, N) or (H, W, N, C)
        
        # for larger sizes, this might be required
        del nbr_rows
        del nbr_cols

        # If constant padding, replace the out-of-bounds values.
        if mode == 'constant':
            gathered_nbrs[out_of_bounds] = const_value
            del out_of_bounds

        # 3. --- Apply Kernel and Aggregate ---
        kernel = self.window.kernel.patch # Shape: (N,)

        # Apply the kernel using direct, robust broadcasting
        if image.ndim == 3: # Color image
            # Reshape kernel from (N,) to (1, 1, N, 1) for broadcasting
            kernel_reshaped = kernel.reshape(1, 1, -1, 1)
            # (H, W, N, C) * (1, 1, N, 1) -> (H, W, N, C)
            weighted_sum = np.sum(gathered_nbrs * kernel_reshaped, axis=2)
        else: # Grayscale image
            # No need to reshape kernel for grayscale.
            # (H, W, N) * (N,) -> (H, W, N)
            weighted_sum = np.sum(gathered_nbrs * kernel, axis=2)

        return weighted_sum


# ------------------------------------------------------------------ #
# ------------- 2. Kernel Based EdgeDetect Filters ----------------- #
# ------------------------------------------------------------------ #

# ------------- 2.1. Laplacian Edge detection ---------------------- #
class laplacian_edge_detection():
    '''
    The goal of this filter is to take a binary image and 
    given a window, will do Laplacian edge detection on the image
    '''
    def __init__(self, n: int, var: float):
        '''
        **Inputs**
        ----
        - n [int]: sqaure window size (>= 1)
        
        **Outputs**
        ----    
        None

        **Description**
        ----
        creates a window ie a set of relative coordinates given n.
        The handling of odd and even windows is different.
        '''
        self.window= square_window(n, kernel= laplacian_kernel(n))
    
    def apply(self, image, mode='constant', const_value= 0):
        '''
        **Inputs**
        ----
        - image [np.ndarray]: image which is to be operated on, (H,W) or (H,W,C) is fine as well
        - mode [str]: padding mode, whether replicate padding or constant value padding.
        - const_value [float | np.ndarray]
        
        **Outputs**
        ----    
        - final_img [np.ndarray]: final image after operating on it of same dims as input image

        **Description**
        ----
        Takes in an image (can be rgb, grayscale or binary) and outputs the laplacian edge image
        '''
        assert image is not None, "Image matrix cannot be \33[1mNone\33[0m!"
        assert isinstance(image,(np.ndarray,)) and len(image.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
        assert isinstance(const_value, (float, int, np.ndarray)), "\033[1mconst_value\33[0m must be a numerical type or a numpy.ndarray with dim. 1"

        # 1. --- Extract Neighbor Coordinates ---
        H, W = image.shape[:2]
        nbr_rows, nbr_cols = self.window.extract_nbrs((H, W)) # Shape: (H, W, N)

        # 2. --- Gather Pixels with Boundary Handling ---
        # The common logic is now outside the 'if' statements.
        
        # First, identify out-of-bounds locations (only needed for 'constant' mode)
        if mode == 'constant':
            out_of_bounds = (nbr_rows < 0) | (nbr_rows >= H) | \
                            (nbr_cols < 0) | (nbr_cols >= W)

        # Clip indices to prevent errors. This is safe for both modes.
        np.clip(nbr_rows, 0, H - 1, out=nbr_rows)
        np.clip(nbr_cols, 0, W - 1, out=nbr_cols)
        
        # Gather all neighbors in one go. Works for both grayscale and color.
        gathered_nbrs = image[nbr_rows, nbr_cols] # Shape: (H, W, N) or (H, W, N, C)
        
        # for larger sizes, this might be required
        del nbr_rows
        del nbr_cols

        # If constant padding, replace the out-of-bounds values.
        if mode == 'constant':
            gathered_nbrs[out_of_bounds] = const_value
            del out_of_bounds

        # 3. --- Apply Kernel and Aggregate ---
        kernel = self.window.kernel.patch # Shape: (N,)

        # Apply the kernel using direct, robust broadcasting
        if image.ndim == 3: # Color image
            # Reshape kernel from (N,) to (1, 1, N, 1) for broadcasting
            kernel_reshaped = kernel.reshape(1, 1, -1, 1)
            # (H, W, N, C) * (1, 1, N, 1) -> (H, W, N, C)
            weighted_sum = np.sum(gathered_nbrs * kernel_reshaped, axis=2)
        else: # Grayscale image
            # No need to reshape kernel for grayscale.
            # (H, W, N) * (N,) -> (H, W, N)
            weighted_sum = np.sum(gathered_nbrs * kernel, axis=2)

        return weighted_sum
    

# ------------------------------------------------------------------ #
# ------------- 3. Kernel Based Sharpening Filters ----------------- #
# ------------------------------------------------------------------ #

# ------------- 3.1 Unsharp Masking -------------------------------- #
class unsharp_masking_filter(Filter):
    '''
    The goal of this filter is to take a binary image and 
    given a window, will do unsharp masking on image
    '''
    def __init__(self, n: int, var: float):
        '''
        **Inputs**
        ----
        - n [int]: sqaure window size (>= 1)
        - var [float]: variance
        
        **Outputs**
        ----    
        None

        **Description**
        ----
        creates a window ie a set of relative coordinates given n.
        The handling of odd and even windows is different.
        '''
        self.gaussian= gaussian_filter(n, var)
    
    def apply(self, image, p, mode='constant', const_value= 0):
        '''
        **Inputs**
        ----
        - image [np.ndarray]: image which is to be operated on, (H,W) or (H,W,C) is fine as well
        - p [float]: a value between 0 and 1 which controls the amount of sharpness
        - mode [str]: padding mode, whether replicate padding or constant value padding.
        - const_value [float | np.ndarray]
        
        **Outputs**
        ----    
        - final_img [np.ndarray]: final sharpened image, of same dims as input image

        **Description**
        ----
        Takes in an image (can be rgb, grayscale or binary) and outputs sharpened image
        '''
        assert image is not None, "Image matrix cannot be \33[1mNone\33[0m!"
        assert isinstance(image,(np.ndarray,)) and len(image.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
        assert isinstance(const_value, (float, int, np.ndarray)), "\033[1mconst_value\33[0m must be a numerical type or a numpy.ndarray with dim. 1"

        I_blur= self.g(image, mode, const_value)
        I_sharp= image + p*(image - I_blur) if image.ndim > 2 \
                else image + p*(image - I_blur.squeeze())
        return np.clip(I_sharp, 0, K-1)

# ---------------- 3.1. Laplacian Based Sharpening ----------------------------- #
class laplacian_sharping_filter(Filter):
    '''
    The goal of this filter is to take a binary image and 
    given a window, will do unsharp masking on image
    '''
    def __init__(self, n: int, var: float):
        '''
        **Inputs**
        ----
        - n [int]: sqaure window size (>= 1)
        - var [float]: variance
        
        **Outputs**
        ----    
        None

        **Description**
        ----
        creates a window ie a set of relative coordinates given n.
        The handling of odd and even windows is different.
        '''
        self.gaussian= gaussian_filter(n, var)
    
    def apply(self, image, p, mode='constant', const_value= 0):
        '''
        **Inputs**
        ----
        - image [np.ndarray]: image which is to be operated on, (H,W) or (H,W,C) is fine as well
        - p [float]: a value between 0 and 1 which controls the amount of sharpness
        - mode [str]: padding mode, whether replicate padding or constant value padding.
        - const_value [float | np.ndarray]
        
        **Outputs**
        ----    
        - final_img [np.ndarray]: final sharpened image, of same dims as input image

        **Description**
        ----
        Takes in an image (can be rgb, grayscale or binary) and outputs sharpened image
        '''
        assert image is not None, "Image matrix cannot be \33[1mNone\33[0m!"
        assert isinstance(image,(np.ndarray,)) and len(image.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
        assert isinstance(const_value, (float, int, np.ndarray)), "\033[1mconst_value\33[0m must be a numerical type or a numpy.ndarray with dim. 1"

        I_blur= self.g(image, mode, const_value)
        I_sharp= image + p*(image - I_blur) if image.ndim > 2 \
                else image + p*(image - I_blur.squeeze())
        return np.clip(I_sharp, 0, K-1)