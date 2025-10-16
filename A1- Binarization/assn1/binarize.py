from assn1.constants import *
from assn1.histogram import histogram
import numpy as np

def binarize(img, t):
    '''
    **Inputs**
    ----
    - img [np.ndarray | List]: Image matrix of size M x N
    - t [int]: an integer between 0,1,...,K-1 (=255), threshold for binarization

    **Outputs**
    ----    
    - img_bin [np.ndarray]: binarized image

    **Description**
    ----
    Binarizes `img` given threshold t
    '''
    ## error checks
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 

    return np.astype(img > t, np.int32)

def binarize0(img, t):
    '''
    **Inputs**
    ----
    - img [np.ndarray | List]: Image matrix of size M x N
    - t [int]: an integer between 0,1,...,K-1 (=255), threshold for binarization

    **Outputs**
    ----    
    - img_bin [np.ndarray]: binarized image

    **Description**
    ----
    Binarizes `img` given threshold t. Instead of 0-1 binarization, -1 1 binarization happens
    less than threshold is
    '''
    ## error checks
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 

    return np.where(img > t, 1, -1)

##
# 2.1. Computing what the assignment asks
##

def within_class_var(img, t):
    '''
    **Inputs**
    ----
    - img [np.ndarray | List]: Image matrix of size M x N
    - t [int]: an integer between 0,1,...,K-1 (=255), threshold for binarization

    **Outputs**
    ----    
    - var_w [np.float64]: within class variance

    **Description**
    ----
    Calculates within class variance of the image for given
    threshold t
    '''
    ## error checks
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 
    
    ## compute the normalized histogram -> O(M*N)
    p= histogram(img, normalized= True)

    ## error check
    assert t>=0 and t< len(p), "Threshold can only take value between minimum and maximum pixel intensity values"

    ## compute the overall mean of image -> O(K)
    mu_T= 0
    for k in range(len(p)):
        mu_T+= k*p[k]
    
    ## calculate class 0 probability and mean -> O(t)
    mu_0, w_0 = 0, 0
    for k in range(t+1):
        w_0 += p[k]
        mu_0+= k*p[k]

    ## calculate class 1 probability and mean -> O(1)
    w_1= 1-w_0
    mu_1 = 0 if np.fabs(w_1) < 1e-8 else (mu_T - w_0*mu_0)/w_1 
    # floating points can vary in bits a little bit and be very close.
    # So, we use a tolerance, but == sign is just as good

    ## calculating the class variance*probability (var_i * w_i)
    v_0, v_1 = 0, 0
    for k in range(t+1):
        v_0 += (k-mu_0)*(k-mu_0)*p[k]
    for k in range(t+1,len(p)):
        v_1 += (k-mu_0)*(k-mu_0)*p[k]
    
    return v_0 + v_1

def between_class_var(img, t):
    '''
    **Inputs**
    ----
    - img [np.ndarray]: Image matrix of size M x N
    - t [int]: an integer between 0,1,...,K-1 (=255), threshold for binarization

    **Outputs**
    ----    
    - var_w [np.float64]: between class variance

    **Description**
    ----
    Calculates between class variance of the image for given
    threshold t
    '''
    ## error checks
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 
    
    ## compute the normalized histogram -> O(M*N)
    p= histogram(img, normalized= True)

    ## error check
    assert t>=0 and t< len(p), "Threshold can only take value between minimum and maximum pixel intensity values"

    ## compute the overall mean of image -> O(K)
    mu_T= 0
    for k in range(len(p)):
        mu_T+= k*p[k]
    
    ## calculate class 0 probability and mean -> O(t)
    mu_0, w_0 = 0, 0
    for k in range(t+1):
        w_0 += p[k]
        mu_0+= k*p[k]

    ## calculate class 1 probability and mean -> O(1)
    w_1= 1-w_0
    mu_1 = 0 if np.fabs(w_1) < 1e-8 else (mu_T - w_0*mu_0)/w_1 
    # floating points can vary in bits a little bit and be very close.
    # So, we use a tolerance, but == sign is just as good

    return w_0*(mu_0-mu_T)**2 + w_1*(mu_1 - mu_T)**2

##
# 2.2. Computing what is actually required
##

def otsu_within_class(img):
    '''
    **Inputs**
    ----
    - img [np.ndarray]: Image matrix of size M x N
    
    **Outputs**
    ----    
    - var_w [np.ndarray]: array containing the within class varaince for all thresholds
    - t [int]: optimum threshold for image binarization

    **Description**
    ----
    Gives you optimum threshold by minimizing the within class variances
    
    '''

    ## error check
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 

    ## compute histogram once
    p= histogram(img, normalized= True)

    ## compute overall mean, will be required
    r= np.array(range(len(p)))      # will be used many times later on
    mu_T= np.sum(r*p)               # mean = sum(k*p[k])
    var_T= np.sum(p*(r-mu_T)**2)    # variance= sum(p[k]*(k-mu_T)^2)

    ## maintain running variables for class probs and mean*prob
    w_0, w_1= 0, 1                      # class probs, corresponds to threshold where everything black
    m_0, m_1= 0, mu_T                   # class mean * class prob, corresponds to threshold where everything black
    var_w= np.zeros(len(p))             # within class variance for all t values gonna be filled in here

    ## loop through all thresholds, calculate sigma_w(t), update running variables
    for t in range(len(p)):
        ## update class probs and means     
        w_0+= p[t] 
        w_1= 1-w_0
        m_0+= t*p[t]
        m_1-= t*p[t]

        ## guard against 0/0 division!
        if np.fabs(w_0) < FT or np.fabs(w_1) < FT:
            var_w[t]= var_T     # one of the classes has zero prob, in which case w_0*sigma_0 + w_1*sigma_1 -> total variance
            continue
        mu_0= m_0/w_0 
        mu_1= m_1/w_1
        ## calculating class variances (kind of...) and within class variances
        v_0= np.sum(p[:t+1]*(r[:t+1]-mu_0)**2)  # w_0 * var_0, avoid using for loops
        v_1= np.sum(p[t+1:]*(r[t+1:]-mu_1)**2)  # w_1 * var_1, avoids using for loops
        var_w[t]= v_0 + v_1
    
    ## return the argument that minimizes within class variance
    return var_w, np.argmin(var_w)

def otsu_between_class(img):
    '''
    **Inputs**
    ----
    - img [np.ndarray]: Image matrix of size M x N
    
    **Outputs**
    ----    
    - var_b [np.ndarray]: array containing the between class varaince for all thresholds
    - t [int]: optimum threshold for image binarization

    **Description**
    ----
    Gives you optimum threshold by maximizing the between class variances
    
    '''

    ## error check
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 

    ## compute histogram once
    p= histogram(img, normalized= True)

    ## compute overall mean, will be required
    r= np.array(range(len(p)))      # will be used many times later on
    mu_T= np.sum(r*p)               # mean = sum(k*p[k])
    var_T= np.sum(p*(r-mu_T)**2)    # variance= sum(p[k]*(k-mu_T)^2)

    ## maintain running variables for class probs and mean*prob
    w_0, w_1= 0, 1                      # class probs, corresponds to threshold where everything black
    m_0, m_1= 0, mu_T                   # class mean * class prob, corresponds to threshold where everything black
    var_b= np.zeros(len(p))             # within class variance for all t values gonna be filled in here

    ## loop through all thresholds, calculate sigma_w(t), update running variables
    for t in range(len(p)):
        ## update class probs and means     
        w_0+= p[t] 
        w_1= 1-w_0
        m_0+= t*p[t]
        m_1-= t*p[t]

        ## guard against 0/0 division!
        if np.fabs(w_0) < FT or np.fabs(w_1) < FT:
            var_b[t]= 0         # one of the classes has zero prob, in which case w_0*w_1*(mu_1 - mu_0)^2 = 0
            continue
        mu_0= m_0/w_0 
        mu_1= m_1/w_1
        ## calculating between class variance for t
        var_b[t]= w_0*w_1*(mu_0 - mu_1)**2
    
    ## return the argument that minimizes within class variance
    return var_b, np.argmax(var_b)

def adaptive_otsu(img, block_size: tuple, overlap= 0.2):
    '''
    **Inputs**
    ----
    - img [np.ndarray | List]: Image matrix of size M x N
    - block_size [tuple]: tuple of length 2 which denotes the block size
    - overlap [float]: overlap %

    **Outputs**
    ----    
    - img_bin [np.ndarray]: binarized image 0-1 binarized

    **Description**
    ----
    Binarizes `img` optimally binarized using otsu algorithm
    '''
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) == 2, "\33[1mimg\33[0m must be a 2D numpy.ndarray" 
    assert isinstance(block_size, (tuple, list)) and len(block_size)== 2, "\33[1mblock_size33[0m must be a tuple or list of length 2"
    assert block_size[0] <= img.shape[0] and block_size[1] <= img.shape[1], "Block dimensions should not exceed image dimensions"

    ## calculate strides
    block_h, block_w= block_size
    img_h, img_w= img.shape

    ## intialize vote_counter (stores how many blocks contributed to each pixel)
    vote_count= np.zeros_like(img, dtype= float)

    ## calculate strides
    stride_h= max(1,int(block_h*(1-overlap)))   # ensure stride is atleast 1
    stride_w= max(1,int(block_w*(1-overlap)))

    ## block wise binarization
    for i in range(0, img_h, stride_h):
        for j in range(0, img_w, stride_w):
            ## extract current block
            blk_end_col= min(j+block_w-1, img_w-1)
            blk_end_row= min(i+block_h-1, img_h-1)

            block= img[i:blk_end_row+1, j:blk_end_col+1]

            ## binarize that block, -1, 1 for majority related ops
            _, t_opt= otsu_between_class(block)
            bin0_block= binarize0(block, t_opt)

            vote_count[i:blk_end_row+1, j:blk_end_col+1] += bin0_block
    
    ## reconstruct the image
    res= np.where(vote_count > 0, 1, 0)

    return res







        




    