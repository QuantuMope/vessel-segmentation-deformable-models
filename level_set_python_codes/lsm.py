"""
	Main function for running a level set method to do image segmentation.
	The algorithm is based on a paper "Retina Image Vessel Segmentation Using a Hybrid CGLI Level Set Method" by Chen et al.
	Note: I replaced the Gaussian convolution with a heat equation.
"""

# Import libararies
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct # for Fourier transforms
from nptyping import NDArray, Float64
from scipy.ndimage.filters import gaussian_filter # For a Gaussian blur

# Level set method class
class LSM:
    def __init__(self, n1, n2, rho, sigma, omega, epsilon):
        self.n1 = n1
        self.n2 = n2
        self.rho = rho
        self.sigma = sigma
        self.omega = omega
        self.epsilon = epsilon
        self.phi = np.zeros((self.n2, self.n1))
        self.phi_x = np.zeros_like(self.phi)
        self.phi_y = np.zeros_like(self.phi)
        self.phi_1norm = np.zeros_like(self.phi)

        self.f1 = np.zeros_like(self.phi)
        self.f2 = np.zeros_like(self.phi)

        self.Spf = np.zeros_like(self.phi)

        self.c1 = 1 # default value
        self.c2 = 1 # default value

        jj = np.arange(n1)
        ii = np.arange(n2)
        ii, jj = np.meshgrid(ii,jj)

        self.kernel = 2.0*n1*n1*(1.0 - np.cos(np.pi*jj/n1)) + 2.0*n2*n2*(1.0 - np.cos(np.pi*ii/n2))
    
    # implement 2D DCT
    def dct2(self, a):
        return dct(dct(a.T, norm='ortho').T, norm='ortho')
    # implement 2D IDCT
    def idct2(self, a):
        return idct(idct(a.T, norm='ortho').T, norm='ortho')
    
    # initialize phi with a quadratic function
    def initialize_phi(self, image: NDArray[Float64]):
        x = np.linspace(0.5/self.n1,1-0.5/self.n1,self.n1)
        y = np.linspace(0.5/self.n2,1-0.5/self.n2,self.n2)
        x, y = np.meshgrid(x,y)
        
        self.phi = - (x-0.5)**2 - (y-0.8)**2 + 0.3**2
        
        
    """
        Heat equation
    """
    def conv_heat(self, A: NDArray[Float64], sigma: Float64) -> NDArray[Float64]:
        A_f = self.dct2(A)
        A_f *= np.exp(-2*sigma*self.kernel)
        return self.idct2(A_f)
        
    """
        Gaussian blur
    """
    def conv_gaussian(self, A: NDArray[Float64], sigma: Float64) -> NDArray[Float64]:
        return gaussian_filter(A, sigma=sigma)
        
    """
        Apply convolution on phi.
        I am using a heat equation with FFT.
        Modify the code to implement a different convolution.
        
        Given numpy 2d array
        Returns numpy 2d array
    """
    def conv(self, A: NDArray[Float64], sigma = -1) -> NDArray[Float64]:
        if sigma == -1:
            sigma = self.sigma
            
        return self.conv_heat(A, sigma) # to use a heat equation
#         return self.conv_gaussian(A, sigma) # to use a Gaussian kernel
        
    """
        compute (partial_x phi) and (partial_y phi)
        computed values: self.phi_x, self.phi_y
    """
    def compute_absolute_gradient(self):
        phi_x1 = np.power(self.phi[:,:-1] - self.phi[:,1:],2)
        phi_y1 = np.power(self.phi[:-1,:] - self.phi[1:,:],2)
        
        self.phi_x[:,:-1] = phi_x1
        self.phi_y[:-1,:] = phi_y1
        
        self.phi_1norm = np.sqrt(self.phi_x + self.phi_y)
    
    """
        Compute H_epsilon(phi)
    """
    def H_epsilon(self, z: NDArray[Float64]) -> NDArray[Float64]:
        return 0.5 * (1 + 2 * np.arctan(z/self.epsilon) / np.pi)
    
    """
        Compute gL and return 2d numpy array
    """
    def compute_gL(self, image: NDArray[Float64]):
        tmp = image - 0.5 * (self.f1 + self.f2)
        tmp = self.conv(tmp)
        
        tmp2 = np.max(np.abs(tmp))
        
        return tmp / tmp2
    
    """
        Compute gL and return 2d numpy array
    """
    def compute_gG(self, image: NDArray[Float64]):
        tmp = image - 0.5 * (self.c1 + self.c2)
        tmp2 = np.max(np.abs(tmp))
        
        return tmp / tmp2
    
    """
        Given image (2d numpy array)
        Returns f1 and f2 (2d numpy array)
    """
    def compute_local_bin_value(self, image: NDArray[Float64]):
        self.tmp = self.H_epsilon(self.phi)
        self.f1  = image * self.tmp
        self.f2  = image - self.f1
        
        # compute a denominator (tmp) and a numerator (self.f1)
        # from equation (10)
        self.tmp = self.conv(self.tmp)
        self.f1  = self.conv(self.f1)
        self.f2 = self.conv(self.f2)
        
        # compute f1 numerator/denominator from equation (10)
        self.f1 = self.f1/self.tmp
        
        # compute f2 numerator/denominator from equation (10)
        self.f2 = self.f2 / (1 - self.tmp)
        
    """
        Update c1 and c2
        From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4429514/.
    """
    def compute_the_region_average_intensity(self, image: NDArray[Float64]):
        # computing c1
        self.tmp[:,:] = self.H_epsilon(self.phi)
        denom = np.sum(self.tmp)
        numer = np.sum(image * self.tmp)
        
        self.c1 = numer/denom
        
        # computing c2
        denom = np.sum(1 - self.tmp)
        numer = np.sum(image * (1 - self.tmp))
        
        self.c2 = numer/denom
        
    """
        Compute the Spf from gL and gG
    """
    def compute_the_local_global_force(self, image: NDArray[Float64]):
        self.Spf = self.compute_gL(image)
        self.tmp = self.compute_gG(image)
        
        self.Spf += self.omega * self.tmp
        
    """ 
        update phi with a given step size
    """
    def update_phi(self, step_size: Float64, image):
        self.phi -= step_size * (self.Spf * self.phi_1norm)
