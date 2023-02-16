

import numpy as np
from scipy.ndimage import filters
from scipy.signal.windows import gaussian
import torch
def gaussian_kernel_smoother(y, sigma, window):
    b = gaussian(window, sigma)
    y_smooth = np.zeros(y.shape)
    neurons = y[:, 0].size
    for neuron in range(neurons):
        y_smooth[neuron, :] = filters.convolve1d(y[neuron, :], b)/b.sum()
    return y_smooth



def get_diagonal(matrix):
    output=torch.zeros((matrix.shape[0],matrix.shape[1]))
    for ii in range( matrix.shape[0]):
        output[ii,:] = torch.diagonal(torch.squeeze(matrix[ii]))
    return output

def calDesignMatrix_V2(X,h):
    '''

    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h, X.shape[1]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i, : , :]= (PadX[i:h+i,:])
    return XDsgn

def calSmoothNeuralActivity(data,gausWindowLength,gausWindowSigma):
    x=np.linspace(-1*gausWindowSigma,1*gausWindowSigma,gausWindowLength)
    gausWindow=1/(2*np.pi*gausWindowSigma)*np.exp(-0.5*(x**2/gausWindowSigma**2))
    gausWindow=gausWindow/np.max(gausWindow)
    #plt.plot(x,gausWindow)
    #plt.show()
    dataSmooth=np.zeros(data.shape)
    for i in range(data.shape[1]):
        dataSmooth[:,i]=np.convolve(data[:,i],gausWindow,'same')
        #dataSmooth[np.where(dataSmooth[:,i] <0), i]=0
    #plt.subplot(2,1,1)
    #plt.plot(data[:5000,1])
    #plt.subplot(2, 1, 2)
    #plt.plot(dataSmooth[:5000, 1])
    #plt.show()
    return dataSmooth
def calInformetiveChan(data,minNumSpiks):
    return np.where(np.sum(data,axis=0)>minNumSpiks)

def calDesignMatrix_V2(X,h):
    '''

    :param X: [samples*Feature]
    :param h: hist
    :return: [samples*hist*Feature]

    '''
    PadX = np.zeros([h , X.shape[1]])
    PadX =np.concatenate([PadX,X],axis=0)
    XDsgn=np.zeros([X.shape[0], h, X.shape[1]])
    # print(PadX.shapepe)
    for i in range(0,XDsgn.shape[0]):
         #print(i)
         XDsgn[i, : , :]= (PadX[i:h+i,:])
    return XDsgn

def calSmoothNeuralActivity(data,gausWindowLength,gausWindowSigma):
    x=np.linspace(-1*gausWindowSigma,1*gausWindowSigma,gausWindowLength)
    gausWindow=1/(2*np.pi*gausWindowSigma)*np.exp(-0.5*(x**2/gausWindowSigma**2))
    gausWindow=gausWindow/np.max(gausWindow)
    #plt.plot(x,gausWindow)
    #plt.show()
    dataSmooth=np.zeros(data.shape)
    for i in range(data.shape[1]):
        dataSmooth[:,i]=np.convolve(data[:,i],gausWindow,'same')
        #dataSmooth[np.where(dataSmooth[:,i] <0), i]=0
    #plt.subplot(2,1,1)
    #plt.plot(data[:5000,1])
    #plt.subplot(2, 1, 2)
    #plt.plot(dataSmooth[:5000, 1])
    #plt.show()
    return dataSmooth
def calInformetiveChan(data,minNumSpiks):
    return np.where(np.sum(data,axis=0)>minNumSpiks)

def get_normalized(x, config):
    if config['supervised']:
        return (x-x.mean())/x.std()
    else:
        return  (x-x.mean())/x.std()