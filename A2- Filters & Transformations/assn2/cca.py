from .constants import *
from .binarize import otsu_between_class, adaptive_otsu
from .histogram import histogram
from .binarize import binarize
from .dsu import DSU
import numpy as np

def find_connected_components_4nbr(bin_img):
    '''
    **Inputs**
    ----
    - bin_img [np.ndarray]: Image matrix of size M x N for binarized image
    
    **Outputs**
    ----    
    - R [np.ndarray]: The region matrix with all connected components
    - dsu [DSU]: dsu containing merge resolves

    **Description**
    ----
    Gives the region matrix and the dsu (resolving merge equivalencies)
    
    '''
    ## error check
    assert bin_img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(bin_img,(np.ndarray,)) and len(bin_img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 

    ## zero padding on the left and top edge (for edge pixel purposes)
    img_pad= np.zeros((bin_img.shape[0]+1, bin_img.shape[1]+1))
    img_pad[1:, 1:]= bin_img

    ## intialize a DSU to resolve merge equivalences
    dsu= DSU()

    ## region array (zero initialized)
    R= np.zeros_like(img_pad, dtype=int)

    ## component counter (any zero region corresponds to black pixels)
    k= 1

    ## first pass (finds CCs)
    for i in range(1,img_pad.shape[0]):
        for j in range(1,img_pad.shape[1]):
            if img_pad[i,j]== 0: continue
            else:
                top_label= int(R[i-1, j])
                left_label= int(R[i, j-1])

                if top_label== 0 and left_label== 0:
                    ## both top and left are useless, new region found
                    R[i,j]= k
                    dsu.make_set(k)
                    k+=1
                elif top_label== 0 and left_label!= 0:
                    ## left pixel is part of something
                    R[i,j]= R[i,j-1]
                    dsu.size[dsu.find_set(left_label)]+=1
                elif top_label!= 0 and left_label== 0:
                    ## top pixel is part of something
                    R[i,j]= R[i-1,j]
                    dsu.size[dsu.find_set(top_label)]+=1
                else:
                    # Case D: Both neighbors have labels, may need merging
                    min_label = min(top_label, left_label)
                    R[i, j] = min_label
                    
                    # If the neighbors belong to different sets, union them
                    if dsu.find_set(top_label) != dsu.find_set(left_label):
                        dsu.union_set(top_label, left_label)
                    
                    # Increment size of root
                    final_root = dsu.find_set(min_label)
                    dsu.size[final_root] += 1
    
    ## check if image has some foreground at all
    largest_cc, cc_size= dsu.get_largest_component()
    if largest_cc== 0:
        print("No foreground components at all!")
        return np.zeros((bin_img.shape[0], bin_img.shape[1]), dtype=np.int32), dsu

    ## second pass (relabel all the things)
    for i in range(1, img_pad.shape[0]):
        for j in range(1, img_pad.shape[1]):
            if R[i,j] > 0:
                root = dsu.find_set(R[i, j])
                R[i, j] = root

    return R[1:, 1:], dsu

def find_connected_components_8nbr(bin_img: np.ndarray):
    '''
    **Inputs**
    ----
    - bin_img [np.ndarray]: Image matrix of size M x N for binarized image
    
    **Outputs**
    ----    
    - R [np.ndarray]: The region matrix with all connected components
    - dsu [DSU]: dsu containing merge resolves

    **Description**
    ----
    Gives the region matrix and the dsu (resolving merge equivalencies)
    
    '''
    ## error check
    assert bin_img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(bin_img,(np.ndarray,)) and len(bin_img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 

    ## zero padding on the left and top edge (for edge pixel purposes)
    img_pad= np.zeros((bin_img.shape[0]+1, bin_img.shape[1]+1))
    img_pad[1:, 1:]= bin_img

    h, w= img_pad.shape

    ## intialize a DSU to resolve merge equivalences
    dsu= DSU()

    ## region array (zero initialized)
    R= np.zeros_like(img_pad, dtype=int)

    ## component counter (any zero region corresponds to black pixels)
    k= 1

    ## these steps are the same as before
    for i in range(1,h):
        for j in range(1, w):
            if img_pad[i,j]==0: continue
            else:
                ## add the non-zero neighbours
                labels= set()
                if R[i-1, j] > 0: labels.add(int(R[i-1, j]))
                if R[i, j-1] > 0: labels.add(int(R[i, j-1]))
                if R[i-1, j-1] > 0: labels.add(int(R[i-1, j-1]))
                if j+1<w and R[i-1, j+1] >0: labels.add(int(R[i-1, j+1]))

                if not labels:  # no non-zero neighbour => new component
                    R[i,j]= k
                    dsu.make_set(k)
                    k+=1
                else:
                    # find the nonzero labels
                    min_label= min(labels)
                    R[i,j]= min_label

                    for m in labels:
                        for l in labels:
                            dsu.union_set(m, l)
                    
                    dsu.size[dsu.find_set(min_label)]+=1
    
    ## check if image has some foreground at all
    largest_cc, cc_size= dsu.get_largest_component()
    if largest_cc== 0:
        print("No foreground components at all!")
        return np.zeros((bin_img.shape[0], bin_img.shape[1]), dtype=np.int32), dsu

    ## second pass (relabel all the things)
    for i in range(1, img_pad.shape[0]):
        for j in range(1, img_pad.shape[1]):
            if R[i,j] > 0:
                root = dsu.find_set(R[i, j])
                R[i, j] = root

    return R[1:, 1:], dsu

                    
        

def highlight_largest_connected_component_4nbr(img: np.ndarray, color=[255, 0, 0]):
    '''
    **Inputs**
    ----
    - img [np.ndarray]: Image matrix of size M x N in grayscale, 2D array
    - color [list | np.ndarray]: color in which to highlight
    
    **Outputs**
    ----    
    - final_img [np.ndarray]: final image with largest Connected Component (CC) highlighted

    **Description**
    ----
    Gives an RGB image matrix of same height and width as input, but is a 3D array
    
    '''
    ## error check
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 

    ## check if average intesity is white or black
    I_avg= np.mean(img)
    I= img
    # take negative if white background
    if I_avg >= 128:
        I= K-1-I

    ## binarize the image
    _, t_opt = otsu_between_class(I)
    bin_img= binarize(I, t_opt)

    ## find connected compoents
    R, dsu= find_connected_components_4nbr(bin_img)

    ## create final image
    mask= R == dsu.get_largest_component()[0]
    final_img= np.repeat(img[:,:,np.newaxis], 3, axis=2)
    final_img[mask] = color

    return final_img

def highlight_largest_connected_component_8nbr(img: np.ndarray, color=[255, 0, 0]):
    '''
    **Inputs**
    ----
    - img [np.ndarray]: Image matrix of size M x N in grayscale, 2D array
    - color [list | np.ndarray]: color in which to highlight
    
    **Outputs**
    ----    
    - final_img [np.ndarray]: final image with largest Connected Component (CC) highlighted

    **Description**
    ----
    Gives an RGB image matrix of same height and width as input, but is a 3D array
    
    '''
    ## error check
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 

    ## check if average intesity is white or black
    I_avg= np.mean(img)
    I= img
    # take negative if white background
    if I_avg >= 128:
        I= K-1-I

    ## binarize the image
    _, t_opt = otsu_between_class(I)
    bin_img= binarize(I, t_opt)

    ## find connected compoents
    R, dsu= find_connected_components_8nbr(bin_img)

    ## create final image
    mask= R == dsu.get_largest_component()[0]
    final_img= np.repeat(img[:,:,np.newaxis], 3, axis=2)
    final_img[mask] = color

    return final_img






