"""
poisson_solver.py
Fast Poisson Reconstruction in Python
Copyright (c) 2014 Jack Doerner
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

"""
Modified by Akshaya Ganesh Nakshathri
"""

import math
import numpy as np
from numba import njit

@njit
def poisson_solver(gx, gy, boundary_image):
    """
    :param gx: Gradient in x direction
    :param gy: Gradient in y direction
    :param boundary_image: Boundary image intensities
    :return:    (0) reconstructed image using Poisson solver
    """

    # gx, gy and boundary image should be of same size
    # This is derived from Dr. Raskar's matlab code here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Laplacian
    gyy = gy[1:,:-1] - gy[:-1,:-1]
    gxx = gx[:-1,1:] - gx[:-1,:-1]

    f = np.zeros(boundary_image.shape)

    f[:-1,1:] += gxx
    f[1:,:-1] += gyy

    # Boundary image
    boundary = boundary_image.copy()
    boundary[1:-1,1:-1] = 0;

    # subtract boundary points contribution
    f_bp = boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[0:-2,1:-1] + boundary[2:,1:-1] # -4*boundary[1:-1,1:-1] can be removed as it is 0
    f = f[1:-1,1:-1] - f_bp

    # compute discrete sine transform
    tt = my_dst(f)
    fsin = my_dst(tt.T).T

    # compute Eigenvalues
    x = np.zeros(f.shape)
    y = np.zeros(f.shape)
    
    for i in range(0, f.shape[0]):
        x[i,:] = np.arange(1,f.shape[1]+1)
        
    for i in range(0, f.shape[1]):
        y[:,i] = np.arange(1,f.shape[0]+1)
    
    denom = (2*np.cos(math.pi*x/(f.shape[1]+1))-2) + (2*np.cos(math.pi*y/(f.shape[0]+1)) - 2)

    f = fsin/denom

    # compute inverse discrete Sine Transform
    tt = my_idst(f)
    img_tt = my_idst(tt.T).T

    # put solution in inner points; outer points obtained from boundary image
    result = boundary
    result[1:-1,1:-1] = img_tt

    result = np.where(result > 255, 255, result)
    result = np.where(result < 0, 0, result)
    
    return result

@njit
def my_dst(x):
    """
    :param x: input matrix as 2D numpy array
    :return:    (0) Discrete Sine Transform of input as 2D numpy array
    """
    # Perform DST similar to matlab https://nl.mathworks.com/help/pde/ug/dst.html
    # For 2D matrix, perform DST column wise
    N = x.shape[0]

    out = np.zeros(np.shape(x))
    #out2 = np.zeros(np.shape(x))

    for i in range(0, x.shape[1]):
        for k in range(0, N):
            n = np.arange(1, N+1)
            sin_array = np.sin(math.pi*(k+1)*n/(N+1));
            out[k, i] = np.sum(x[:,i]*sin_array)

    return out

@njit
def my_idst(x):
    """
    :param x: input matrix as 2D numpy array
    :return:    (0) Inverse Discrete Sine Transform of input as 2D numpy array
    """
    
    # Perform IDST similar to matlab https://nl.mathworks.com/help/pde/ug/dst.html
    # For 2D matrix, perform IDST column wise

    N = x.shape[0]

    out = np.zeros(np.shape(x))
    #out2 = np.zeros(np.shape(x))

    for i in range(0, x.shape[1]):
        for k in range(0, N):
            n = np.arange(1, N+1)
            sin_array = np.sin(math.pi*(k+1)*n/(N+1));
            out[k, i] = np.sum(x[:,i]*sin_array)

    out = out * 2/(N+1)

    return out