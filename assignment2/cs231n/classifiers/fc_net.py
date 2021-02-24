from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = weight_scale * np.random.randn(input_dim,hidden_dim)  #用标准差为weight_scale的高斯分布初始化参数W,隐藏层的参数矩阵行数是3*32*32，列数是100
        self.params["b1"] = np.zeros((hidden_dim,))                               #偏置B的初始都为0
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim,num_classes)#输出层的参数矩阵行数是100，列数10.
        self.params["b2"] = np.zeros((num_classes,))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']  #赋值     
        out1, cache1 = affine_relu_forward(X, W1, b1)               #计算得分
        scores, cache2 = affine_forward(out1, W2, b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss_data, dscores = softmax_loss(scores, y)                       #softmax 
        dout1, grads['W2'], grads['b2'] = affine_backward(dscores, cache2) #输出到隐藏层
        dx, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache1) #隐藏层到输入
        loss = loss_data + 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))#loss和grads加入正则化
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_bn_ln_relu_dp_forward(x, w, b, gamma, beta, bn_param, normalization, dropout_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    dropout_cache = None
    bn_cache = None
    x_fc, fc_cache = affine_forward(x, w, b)
    if normalization == 'batchnorm':
        x_bn, bn_cache = batchnorm_forward(x_fc, gamma, beta, bn_param)
    elif normalization == 'layernorm':#layer norm
        x_bn, bn_cache = layernorm_forward(x_fc, gamma, beta, bn_param)
    else:
        x_bn = x_fc
    out, relu_cache = relu_forward(x_bn)
    if dropout_param['p'] != 1:
        #print('Using dropout!')
        out, dropout_cache = dropout_forward(out, dropout_param)
    cache = (fc_cache, bn_cache, relu_cache, dropout_cache, normalization)
    return out, cache


def affine_bn_ln_relu_dp_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache, dropout_cache, normalization = cache
    if dropout_cache != None:#dropout
        dout = dropout_backward(dout, dropout_cache)
    dfc = relu_backward(dout, relu_cache)
    if normalization == 'batchnorm':
        dbn, dgamma, dbeta = batchnorm_backward(dfc, bn_cache)
    elif normalization == 'layernorm':#layer norm
        dbn, dgamma, dbeta = layernorm_backward(dfc, bn_cache)
    else:
        dbn = dfc
        dgamma = 1#随便设置无所谓，不能是None
        dbeta = 1
    dx, dw, db = affine_backward(dbn, fc_cache)
    return dx, dw, db, dgamma, dbeta

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #if normalization != None:
        self.params['gamma1'] = np.ones(hidden_dims[0])
        self.params['beta1'] = np.zeros(hidden_dims[0])
          
        num_layers = self.num_layers
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])
        for i in range(2,num_layers):
            self.params['W'+str(i)] = weight_scale * np.random.randn(hidden_dims[i-2], hidden_dims[i-1])
            self.params['b'+str(i)] = np.zeros(hidden_dims[i-1])
            #if normalization != None:
            self.params['gamma'+str(i)] = np.ones(hidden_dims[i-1])
            self.params['beta'+str(i)] = np.zeros(hidden_dims[i-1])
        
        self.params['W'+str(num_layers)] = weight_scale * np.random.randn(hidden_dims[num_layers-2], num_classes)
        self.params['b'+str(num_layers)] = np.zeros(num_classes)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        normalization = self.normalization
        num_layers = self.num_layers
        out, cache = {}, {}
        if self.normalization == None:#参数初始化！
            self.bn_params = [{} for i in range(self.num_layers - 1)]
        if not self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': 1}
                
        out['1'], cache['1'] = affine_bn_ln_relu_dp_forward(X, self.params['W1'], self.params['b1'],
            self.params['gamma1'], self.params['beta1'], self.bn_params[0],
                normalization, self.dropout_param)#下标从1开始
        for i in range(2,num_layers):#最后一层只有全连接没有relu
            out[str(i)], cache[str(i)] = affine_bn_ln_relu_dp_forward(out[str(i-1)], self.params['W'+str(i)], self.params['b'+str(i)],
                self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1],
                    normalization, self.dropout_param)
        
        scores, cache[str(num_layers)] = affine_forward(out[str(num_layers-1)], self.params['W'+str(num_layers)], self.params['b'+str(num_layers)])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dout = {}
        reg_loss = 0
        reg = self.reg
        
        data_loss, dscores = softmax_loss(scores, y)
        dout[str(num_layers)], grads['W'+str(num_layers)], grads['b'+str(num_layers)] = affine_backward(dscores, cache[str(num_layers)])
        
        for i in range(2,num_layers)[::-1]:#[::-1]坐标倒序
            dout[str(i)], grads['W'+str(i)], grads['b'+str(i)], grads['gamma'+str(i)], grads['beta'+str(i)] = affine_bn_ln_relu_dp_backward(dout[str(i+1)], cache[str(i)])
        
        _, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = affine_bn_ln_relu_dp_backward(dout['2'], cache['1'])#第一层
        
        for key, W in self.params.items():
            reg_loss += 0.5 * reg * np.sum(W**2)
            grads[key] += reg * W
        loss = data_loss + reg_loss

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

