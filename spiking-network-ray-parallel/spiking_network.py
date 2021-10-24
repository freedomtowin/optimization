import numpy as np

class SpikingNetwork:
    
    def __init__(self):
        self.forward_list = []
        self._model_shapes = []
        self._layer_numbers = []
        self._l1_regularization = []
        self._layer_num = 0
        self._number_of_parameters = 0
        
    #correlated each column in X with y, find top lag variables
    def lagValues(self,X,y,input_size,output_size,l1_regularization=0):
            
        num = input_size
        def correlation(x,y):
            result = np.correlate(x, y, mode='same')
            result = result[result.size//2:]
            return result
        
        lag_arr = np.zeros((num,X.shape[1]))

        for i in range(X.shape[1]):
            
            ts1 = y.ravel()
            ts2 = X[:,i]
            
            ac = correlation(ts1,ts2)
            t = np.arange(0,ac.size)

            ac = ac[(t<=20)&(t>=1)]
            t = t[(t<=20)&(t>=1)]

            best_5 = np.argsort(np.abs(ac))[-num:]

            autocorrelation_loc = best_5
            peaks = autocorrelation_loc

            peaks = np.round(peaks,0).astype(np.int)

            lag_arr[:,i] = t[peaks].flatten().astype(np.int)

        return lag_arr.astype(np.int)
    
    def autoregressiveComponents(self,X,y,input_size,output_size,l1_regularization=0):
        self.forward_list.append('ar')
        for i in range(lag_arr.shape[1]):
            #parameters for each lag value
            self._model_shapes.append((lag_arr.shape[0],output_size))
            self._number_of_parameters += lag_arr.shape[0]*output_size
            self._layer_numbers.append(self._layer_num)
            self._l1_regularization.append(l1_regularization)
  
        self._model_shapes.append((output_size,1))
        self._number_of_parameters += output_size
        self._layer_numbers.append(self._layer_num)
        self._l1_regularization.append(l1_regularization)
        self._layer_num+=1

        
    def spikingSparseComponents(self,X,y,input_size,output_size,l1_regularization=0):
        self.forward_list.append('sparse_spike')
        for i in range(lag_arr.shape[1]):
            #parameters for each lag value
            self._model_shapes.append((input_size,output_size))
            self._number_of_parameters += input_size*output_size
            self._layer_numbers.append(self._layer_num)
            self._l1_regularization.append(l1_regularization)
    
        self._layer_num+=1

    
    def feedbackComponents(self,window_size,l1_regularization=0):
        self.forward_list.append('feed')
        self._model_shapes.append((window_size,1))
        self._number_of_parameters += window_size
        self._layer_numbers.append(self._layer_num)
        self._l1_regularization.append(l1_regularization)
        self._layer_num+=1
        
    def hiddenComponents(self,input_size,output_size,l1_regularization=0):
        self.forward_list.append('hidden')
        self._model_shapes.append((input_size,output_size))
        self._number_of_parameters += input_size*output_size
        self._layer_numbers.append(self._layer_num)
        self._l1_regularization.append(l1_regularization)
        
        self._model_shapes.append((output_size,1))
        self._number_of_parameters += output_size
        self._layer_numbers.append(self._layer_num)
        self._l1_regularization.append(0)
        self._layer_num+=1
        
    def movingAverage(self,l1_regularization=0.0):
        self.forward_list.append('maw')
        self._model_shapes.append((1,1))
        self._number_of_parameters += 1
        self._layer_numbers.append(self._layer_num)
        self._l1_regularization.append(l1_regularization)
        self._layer_num+=1
        
    @property
    def model_shapes(self):
        return self._model_shapes
    
    @property
    def number_of_parameters(self):
        return self._number_of_parameters
    
    
def AutoRegressiveLayer(X,w,lag_arr):
    max_lag = np.max(lag_arr)
    mask = max_lag-lag_arr
    for j in range(X.shape[1]):
        if j==0:
            pred=np.sum(X[:,j][mask[:,j]].reshape(1,-1).dot(w[j]))
        else:
            pred+=np.sum(X[:,j][mask[:,j]].reshape(1,-1).dot(w[j]))
    return pred+w[j+1]*np.sum(w[0])

def SparseSpikingLayer(X,w,lag_arr,mem):
    max_lag = np.max(lag_arr)
    mask = max_lag-lag_arr
    D = X.shape[1]
    for j in range(D):
        mem+= np.clip(X[:,j][mask[:,j]].reshape(1,-1).dot(w[j]),-3/D,3/D)
    mem = -mem % 3.0
    pred = np.exp(-(mem.T)**2)

    return pred,mem


def HiddenLayer(pred,w):
    return pred.T.dot(w[0]) + w[1]

def Feedback(pred,error,w):
    return pred+np.sum(error*w[0])
    
def MovingAverageWindow(x,window=2):
    np=numpy
    result = np.zeros(x.shape)
    result[:window] = np.mean(x[:window])
    for i in range(window,x.shape[0]):
        result[i] = np.mean(x[i-window:i])
    return result