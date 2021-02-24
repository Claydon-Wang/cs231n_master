from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. w为（3073，10）的矩阵
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)                              # initialize the gradient as zero, dw初始化为和w一样大小的0矩阵,w在svm程序中初始化为（3073，10）的随机矩阵(数很小)

    # compute the loss and the gradient
    num_classes = W.shape[1]                            #num_classes=10
    num_train = X.shape[0]                              #num_train=49000
    loss = 0.0
    for i in range(num_train):                          #逐个计算每个样本的loss
        scores = X[i].dot(W)                            # X[i]（1,3073）× w（3073，10）=scores[1，10]
        correct_class_score = scores[y[i]]              #赋值syi，记录了类别正确的评分
        for j in range(num_classes):                    #循环计算sum(max(0，sj-syi+1))
            if j == y[i]:                               #不计算相同的那一种评分
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1，计算损失L
            if margin > 0:                               #max函数效果
                loss += margin
                dW[:,j]+=X[i, :]                         #数据分类错误时的梯度,dW=(3073,10)
                dW[:,y[i]]-=X[i, :]                      #数据分类正确时的梯度，所有非正确的累减

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train                                    #÷n张图片
    dW /=num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)                          #正则化，得到结果L
    dW += 2 * reg * W                                    #梯度正则化,因为L中正则化部分为R(w)平方，求导后为2R(w)
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]                            #num_classes=10
    num_train = X.shape[0]                              #num_train=49000
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # compute the loss and the gradient

    scores = X.dot(W)                                   #(49000,3073)*(3073,10) = (49000,10)
    scores_y = scores[range(num_train),y]               #y是一个(1,49000)的矩阵，存所有图片正确标签，scores_y(1xN)得到每一个样本的正确分量的得分
    scores_y = np.reshape(scores_y,(num_train,1))       #把scores_y从(1xN)的矩阵变成(Nx1)
    margin = np.maximum(0,scores - scores_y +1)         #广播使scores所有元素减去正确分量，并与0比大小
    margin[range(num_train),y]=0                        #确保正确标签位为0
    loss += np.sum(margin)/num_train                    #求解loss
    loss += reg * np.sum(W * W) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ds = np.zeros_like(margin)                      #创建一个和margin一样大小的0矩阵
    ds[margin>0]=1                                  #求ds中非标签类且条件成立的项为1，因为创建的0矩阵，所以不成立的为0
    rom_sum = -np.sum(ds,axis=1)                    #列相加，因为成立都为1，即统计条件成立的个数
    ds[range(num_train),y] = rom_sum                #给当前标签类的赋值，数值上等于条件成立的相反数
    dW += np.dot(X.T,ds)/num_train                  #由公式推导，x.T * ds求出dW
    dW += 2 * reg * W                               # 加正则化，同上

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
