import numpy as np
import scipy as sp
import scipy.signal

class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `output` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    Moreover, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        input_grad = module.backward(input, output_grad)
    """
    def __init__ (self):
        self._output = None
        self._input_grad = None
        self.training = True
    
    def forward(self, input):
        self._output = self._compute_output(input)
        return self._output

    def backward(self, input, output_grad):
        self._input_grad = self._compute_input_grad(input, output_grad)
        self._update_parameters_grad(input, output_grad)
        return self._input_grad
    

    def _compute_output(self, input):
        raise NotImplementedError
        

    def _compute_input_grad(self, input, output_grad):
        raise NotImplementedError
    
    def _update_parameters_grad(self, input, output_grad):
        pass
    
    def zero_grad(self): 
        pass
        
    def get_parameters(self):
        return []
        
    def get_parameters_grad(self):
        return []
    
    def train(self):
        self.training = True
    
    def evaluate(self):
        self.training = False
    
    def __repr__(self):
        return "Module"

## 1. Batch normalization
class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0.
        self.moving_variance = 1.

    def _compute_output(self, input):
        # Your code goes here. ################################################
        if self.training:
            среднее = np.mean(input, axis=0)
            дисперсия = np.mean((input - среднее)**2, axis=0)
            output = (input - среднее) / np.sqrt(дисперсия + self.EPS)
            self.moving_mean = self.moving_mean * self.alpha + среднее * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + дисперсия * (1 - self.alpha)
            self._сохранённое_среднее = среднее
            self._сохранённая_дисперсия = дисперсия
        else:
            output = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        return output

    def _compute_input_grad(self, input, output_grad):
        # Your code goes here. ################################################
        if self.training:
            N, D = input.shape
            вход_сред = input - self._сохранённое_среднее
            inv_std = 1.0 / np.sqrt(self._сохранённая_дисперсия + self.EPS)
            grad_input = (1./N)*inv_std*(N*output_grad - np.sum(output_grad, axis=0) 
                                         - вход_сред*(np.sum(output_grad*вход_сред, axis=0)/(self._сохранённая_дисперсия+self.EPS)))
        else:
            grad_input = output_grad / np.sqrt(self.moving_variance + self.EPS)
        return grad_input

    def __repr__(self):
        return "BatchNormalization"

class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def _compute_output(self, input):
        output = input * self.gamma + self.beta
        return output
        
    def _compute_input_grad(self, input, output_grad):
        grad_input = output_grad * self.gamma
        return grad_input
    
    def _update_parameters_grad(self, input, output_grad):
        self.gradBeta = np.sum(output_grad, axis=0)
        self.gradGamma = np.sum(output_grad*input, axis=0)
    
    def zero_grad(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def get_parameters(self):
        return [self.gamma, self.beta]
    
    def get_parameters_grad(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"

## 2. Dropout
class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = []
        
    def _compute_output(self, input):
        # Your code goes here. ################################################
        if self.training:
            self.mask = (np.random.rand(*input.shape) > self.p).astype(input.dtype)
            output = (input * self.mask) / (1.0 - self.p)
        else:
            output = input
        return output
    
    def _compute_input_grad(self, input, output_grad):
        # Your code goes here. ################################################
        if self.training:
            grad_input = (output_grad * self.mask) / (1.0 - self.p)
        else:
            grad_input = output_grad
        return grad_input
        
    def __repr__(self):
        return "Dropout"


## 3. Conv2d
import skimage

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size
       
        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def _compute_output(self, input):
        # YOUR CODE ##############################
        pad_size = self.kernel_size // 2
        вход_пад = np.pad(input, ((0,0),(0,0),(pad_size,pad_size),(pad_size,pad_size)), 
                          mode='constant', constant_values=0)
        N, IC, H, W = input.shape
        self._output = np.zeros((N, self.out_channels, H, W))
        for n in range(N):
            for oc in range(self.out_channels):
                сумма = 0
                for ic in range(IC):
                    сумма += sp.signal.correlate(вход_пад[n, ic], self.W[oc, ic], mode='valid')
                self._output[n, oc] = сумма + self.b[oc]
        return self._output
    
    def _compute_input_grad(self, input, gradOutput):
        # YOUR CODE ##############################
        pad_size = self.kernel_size // 2
        N, IC, H, W = input.shape
        _, OC, _, _ = gradOutput.shape
        вход_гр = np.zeros_like(input)
        grad_out_pad = np.pad(gradOutput, ((0,0),(0,0),(pad_size,pad_size),(pad_size,pad_size)), 
                              mode='constant', constant_values=0)
        for n in range(N):
            for ic in range(IC):
                сумма = 0
                for oc in range(OC):
                    w_flip = self.W[oc, ic, ::-1, ::-1]
                    сумма += sp.signal.correlate(grad_out_pad[n, oc], w_flip, mode='valid')
                вход_гр[n, ic] = сумма
        return вход_гр
    
    def accGradParameters(self, input, gradOutput):
        # YOUR CODE #############
        pad_size = self.kernel_size // 2
        N, IC, H, W = input.shape
        вход_пад = np.pad(input, ((0,0),(0,0),(pad_size,pad_size),(pad_size,pad_size)), 
                          mode='constant', constant_values=0)
        self.gradW.fill(0)
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                сумма = 0
                for n in range(N):
                    сумма += sp.signal.correlate(вход_пад[n, ic], gradOutput[n, oc], mode='valid')
                self.gradW[oc, ic] = сумма
        self.gradb = np.sum(gradOutput, axis=(0,2,3))
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' %(s[1],s[0])
        return q
