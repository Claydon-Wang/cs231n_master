from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3073, 10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (49000, 3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means (49000, )
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)   #(3073, 10)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]                                                              #num_classes=10
    num_train = X.shape[0]                                                                #num_train=49000
    scores = X.dot(W)                                                                     #(49000,3073)*(3073,10) = (49000,10)
    scores = np.exp(scores - np.max(scores,axis = 1 , keepdims=True))                     #-去最大值，防止score大时求对数数值爆炸，再指数化
    loss = np.sum(-np.log(scores[range(num_train), y]/np.sum(scores, axis=1)))/num_train  #取正确标签的那一项，先归一化再取-log，最后总和/N
    loss += reg * np.sum(W * W)
    for i in range(num_train):
        dW[:,y[i]] -= X[i]                                                                #先对正确标签类-syi求导，将X矩阵第i行的3073个数存入dW矩阵第y[i]列，相当于x已经经过转置
        for j in range(num_classes):
            dW[:,j] += X[i] * scores[i,j] / np.sum(scores[i])                             #再对所有dsj求导
    dW /= num_train                                                                       #/N
    dW += 2 * reg * W                                                                       #+正则项求导

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]                                                              #num_classes=10
    num_train = X.shape[0]                                                                #num_train=49000
    scores = X.dot(W)                                                                     #(49000,3073)*(3073,10) = (49000,10)
    scores = np.exp(scores - np.max(scores,axis = 1 , keepdims=True))                     #-去最大值，防止score大时求对数数值爆炸，再指数化
    loss = np.sum(-np.log(scores[range(num_train), y]/np.sum(scores, axis=1)))/num_train  #取正确标签的那一项，先归一化再取-log，最后总和/N
    loss += reg * np.sum(W * W)                                                           #加正则项
    ds = scores / np.sum(scores,axis=1,keepdims=True)                                     #对每一项/所在行的总和（当前图片的分数和）
    ds[range(num_train),y] -=1                                                            #每一张图片正确标签项-1（syi求导）
    dW = X.T.dot(ds)                                                                      #求解dw
    dW /= num_train                                                                       #/N
    dW += 2 * reg * W                                                                     #+正则项求导

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
