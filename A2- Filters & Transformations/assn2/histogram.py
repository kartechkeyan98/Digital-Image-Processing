import numpy as np
import matplotlib.pyplot as plt
from .constants import K

def histogram(img: np.ndarray, normalized= False):
    '''
    **Inputs**
    ----
    - img [np.ndarray]: Image of size M x N
    - normalized [bool]: boolean which specifies if histogram is to be normalized

    **Outputs**
    ----
    - hist [np.ndarray]: the histogram of image(size K array)

    **Description**
    ----
    Computes the histogram of an image and returns it.
    Uses tolerance based comparison to avoid artifacts from floating
    point arithmetic.
    
    '''
    ## verify that you get a 2D array
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray"

    ## zero intialized histogram with np.float64 to enable normalization
    hist= np.zeros((K,), dtype= np.float64)

    ## brute force way of doing things
    # M,N= img.shape
    # for i in range(M):
    #     for j in range(N):
    #         hist[img[i,j]]+=1

    ## little bit better way of doing things without relying too much on numpy
    for k in range(K):
        hist[k]= np.sum(np.fabs(img - k) < 1e-6) 
        # Don't need more than 6 decimal precision usually

    return hist if not normalized else hist/img.size

def display_histogram(H, bin_edges= None, title= 'Histogram'):
    '''
    **Inputs**
    ----
    - p [np.ndarray | List]: Histogram of size K computed from any of the above functions
    - bin_edges [np.ndarray | List]: sorted array of size K+1 containing the bin edges of histogram, default is None
    - title [str]: title of the plot, default is 'Histogram'

    **Outputs**
    ----
    None

    **Description**
    ----
    - Displays the histogram using `plt.hist` function
    - `bin_edges` by default will be set to `np.array(range(len(H) + 1))` which is [0, 1,...,K-1, K],
    the edges will be one more than the length of `H`.
    
    '''
    ## error checking
    assert len(bin_edges) == len(H) + 1 if bin_edges is not None else True, "\33[1mbin_edges\33[0m must have size 1 more than \33[1mH\33[0m"

    ## accouting for default bin_edges case 
    if bin_edges is None: bin_edges= np.array(range(len(H) + 1))   

    ## Bin widths and bin centers
    bin_widths= np.diff(bin_edges)              # [edge2 - edge1, edge3 - edge2 ....]
    bin_center= bin_edges[:-1] + bin_widths/2   # basically shifts our edges to centers 

    ## display routine
    plt.bar(bin_center, H, bin_widths, edgecolor='black')
    plt.plot(range(len(H)), H, color= 'red', lw=0.7)

    ## display configs
    plt.title(title)
    plt.xlabel(f'Intensity Level [0 - {len(H)-1}]')
    plt.ylabel('Frequency/Probability')
    plt.grid(True)
    plt.show()
    return

def hist_average(H, bins= None):
    '''
    **Inputs**
    ----
    - p [np.ndarray | List]: Histogram of size K computed from any of the above functions
    - bins [np.ndarray | List]: sorted array of size K bin values (pixel intensities)
    
    **Outputs**
    ----    
    - avg [np.float64]: average intensity of pixels (aka histogram average)

    **Description**
    ----
    Calculates average intensity of the histogram
    '''
    ## error checks
    assert H is not None, "\33[1mH\33[0m cannot be \33[1mNone\33[0m"
    assert len(bins) == len(H) if bins is not None else True, "\33[1mbin_edges\33[0m must have size 1 more than \33[1mH\33[0m"
    
    ## default state of bins
    if bins is None: bins= np.array(range(len(H)))

    ## normalize histogram
    H = H/np.sum(H)

    return np.sum(H*bins)


    