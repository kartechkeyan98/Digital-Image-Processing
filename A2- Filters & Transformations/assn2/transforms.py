from abc import ABC, abstractmethod
import numpy as np
import math

class Transform(ABC):
    '''
    Implements an abstract transform class
    Some methods in this will apply the transform
    completely.
    '''
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, img:np.ndarray, *args, **kwargs):
        pass
    
    def __call__(self, img:np.ndarray, *args, **kwargs):
        return self.apply(img,*args, **kwargs)


class zoom(Transform):
    '''
    Implements zoom which by itself uses bilinear interpolation to 
    calculate values.
    '''
    def __init__(self):
        pass    

    def bilerp(self,img, i, j):
        '''
        **Inputs**
        ----
        - img [np.ndarray]: og image
        - i, j [int | float]: inverse mapped coords

        **Output**:
        ----
        - p [float]: pixel value which is a mixture of surrounding pixels in og img

        **Desciption**
        ----
        Gives the value of (i,j) which might not be integer coords (valid coords)
        for img, but interpolates that. If it lies outside of the image then black
        it out
        '''
        ## 1. manage edge cases and calculate sourrounding coords
        if i < 0 or j < 0 or i >= img.shape[0] or j >= img.shape[1]: return 0
        x, y = math.floor(i), math.floor(j)

        # handles edge cases when x= H-1 or y= W-1 where we may not find x+1, y+1
        x= x if x <= img.shape[0]-2 else img.shape[0]-2
        y= y if y <= img.shape[1]-2 else img.shape[1]-2
        
        ## 2. Use surrounding coords to calculate the interpolated pixel value
        M= np.array(
            [
                [(x+1)*(y+1), -(x+1)*y, -(y+1)*x, x*y],
                [-(y+1), y, y+1, -y],
                [-(x+1), x+1, x, -x],
                [1, -1, -1, 1]
            ]
        )
        I= np.array([img[x,y], img[x,y+1], img[x+1,y], img[x+1,y+1]]).reshape(4,-1)
        A= M@I
        v= np.array([1,i,j,i*j]).reshape(1,-1)
        return (v@A).flatten()

    
    def apply(self, img:np.ndarray, zm: tuple):
        '''
        **Inputs**
        ----
        - img [np.ndarray]: image matrix
        - a [float]: how much the height is magnified
        - b [float]: how much the width is magnified

        **Outputs**
        ----
        mod_img [np.ndarray]: output image

        **Description**
        ----
        This gives the stretched image.
        '''
        assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
        assert isinstance(img,(np.ndarray,)) and len(img.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
        assert isinstance(zm, (tuple)) and len(zm) == 2, "\33[1mzm\33[0m must be 2 length tuple"
        
        a,b = zm
        assert a > 0 and b > 0, "Cannot have magnification factors as negative!"

        ## 1. Calculate the new width and height of transformed image
        H,W= img.shape[:2]
        h1,w1= math.ceil(a*H), math.ceil(b*W)
        assert h1!=0 and w1!=0, "Modified image dimension cannot be zero!"

        # calculate the new shape
        shape_new= list(img.shape)
        shape_new[:2]= [h1,w1]

        # create new image matrix
        mod_img= np.zeros(tuple(shape_new))

        ## 2. Loop through all the pixels and calculate inverse mapping
        # ------ Inverse Mapping: si= i/a, sj= j/b ------------------ #
        for i in range(h1):
            for j in range(w1):
                ## 2.1. calculate source coordinate via inverse mapping
                si, sj= i/a, j/b

                ## 2.2 use bilinear interpolation to calculate the pixel value there
                mod_img[i,j]= self.bilerp(img, si, sj)
        
        return mod_img


class rotate(Transform):
    '''
    Implements rotation which by uses bilinear interpolation to 
    calculate values of inverse mapped coordinates.
    '''
    def __init__(self):
        pass    

    def bilerp(self,img, i, j):
        '''
        **Inputs**
        ----
        - img [np.ndarray]: og image
        - i, j [int | float]: inverse mapped coords

        **Output**
        ----
        - p [float]: pixel value which is a mixture of surrounding pixels in og img

        **Desciption**
        ----
        Gives the value of (i,j) which might not be integer coords (valid coords)
        for img, but interpolates that. If it lies outside of the image then black
        it out
        '''
        ## 1. manage edge cases and calculate sourrounding coords
        if i < 0 or j < 0 or i >= img.shape[0] or j >= img.shape[1]: return 0
        x, y = math.floor(i), math.floor(j)

        # handles edge cases when x= H-1 or y= W-1 where we may not find x+1, y+1
        x= x if x <= img.shape[0]-2 else img.shape[0]-2
        y= y if y <= img.shape[1]-2 else img.shape[1]-2
        
        ## 2. Use surrounding coords to calculate the interpolated pixel value
        M= np.array(
            [
                [(x+1)*(y+1), -(x+1)*y, -(y+1)*x, x*y],
                [-(y+1), y, y+1, -y],
                [-(x+1), x+1, x, -x],
                [1, -1, -1, 1]
            ]
        )
        I= np.array([img[x,y], img[x,y+1], img[x+1,y], img[x+1,y+1]]).reshape(4,-1)
        A= M@I
        v= np.array([1,i,j,i*j]).reshape(1,-1)
        return (v@A).flatten() 


    def apply(self, img, angle: float):
        '''
        **Inputs**:
        - img [np.ndarray]: og image
        - angle [flaot]: angle to rotate anticlockwise in degrees

        **Output**:
        - mod_img [np.ndarray]: rotated image

        **Desciption**:
        Gives rotated image
        '''
        assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
        assert isinstance(img,(np.ndarray,)) and len(img.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
        assert isinstance(angle, (float, int)), "\33[1mangle\33[0m must be numerical type"

        ## 1. Calculate the new size of image
        H,W= img.shape[:2]
        rads= np.radians(angle)
        h1= math.ceil(H*np.fabs(np.cos(rads)) + W*np.fabs(np.sin(rads)))
        w1= math.ceil(W*np.fabs(np.cos(rads)) + H*np.fabs(np.sin(rads)))
        new_shape=list (img.shape)
        new_shape[:2]=[h1,w1]

        mod_img= np.zeros(tuple(new_shape))

        ## 2. For loop through the inverse mapping
        cx, cy= h1/2, w1/2
        for i in range(h1):
            for j in range(w1):
                ## 2.1 inverse mapping
                # center
                x,y= i-cx, j-cy
                # inverse rotate
                si= x*np.cos(rads) + y*np.sin(rads)
                sj= y*np.cos(rads) - x*np.sin(rads)
                # recenter to source coordinates
                si+= H/2
                sj+= W/2
                ## 2.2 linear interpolation (or anything else)
                mod_img[i,j]= self.bilerp(img, si, sj)
        
        return mod_img
    

        
