#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x /= np.linalg.norm(x, axis=1).reshape(x.shape[0], 1)
    ### END YOUR CODE

    return x


def t_normalize_rows():
    print ("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print (x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ()


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models
    # 注意这里是对一个单词的损失和梯度计算
    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)# 输入词向量
    target -- integer, the index of the target word #目标分类
    outputVectors -- "output" vectors (as rows) for all tokens #全词表输出词向量映射
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    # predicted : (3, )
    # outputVectors: (5, 3)
    # target : scala
    D = predicted.shape[0] # 3个神经元
    predicted = predicted.reshape(D, 1) #(3, 1)
    out = outputVectors.dot(predicted) #(5, 1)
    correct_class_score = out[target]
    exp_sum = np.sum(np.exp(out))
    cost = np.log(exp_sum) - correct_class_score

    margin = np.exp(out) / exp_sum # softmax value (5, 1)
    margin[target] += -1
    gradPred = margin.T.dot(outputVectors) # (1, 5) * (5, 3)= (1, 3)
    grad = margin.dot(predicted.T) # (5, 1) * (3, 1).T = (5, 3)

    # print("predicted:", predicted.shape)
    # print("target", target.shape)
    # print("outputVectors", outputVectors.shape)
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    grad = np.zeros(outputVectors.shape) #(5, 3)
    gradPred = np.zeros(predicted.shape)

    out1 = sigmoid(outputVectors[target].dot(predicted))
    cost = - np.log(out1)
    grad[target] += (out1 - 1) * predicted
    gradPred += (out1 - 1) * outputVectors[target]
    for k in range(K):
        # sigmod(-x) = 1 - sigmod(x)
        out2 = sigmoid(-1 * outputVectors[indices[k+1]].dot(predicted))
        cost += -np.log(out2)
        grad[indices[k+1]] += - (out2 - 1) * predicted
        gradPred += - (out2 - 1) * outputVectors[indices[k+1]]
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word #当前词
    C -- integer, context size #窗口
    contextWords -- list of no more than 2*C strings, the context words #上下文
    tokens -- a dictionary that maps words to their indices in
              the word vector list # 全词表
    inputVectors -- "input" word vectors (as rows) for all tokens #全词表输入词向量映射
    outputVectors -- "output" word vectors (as rows) for all tokens #全词表输出词向量映射
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape) #(5, 3)
    gradOut = np.zeros(outputVectors.shape) #(5, 3)

    ### YOUR CODE HERE
    source = tokens[currentWord]
    predicted = inputVectors[source] #输入词到词向量映射
    for target_word in contextWords:
        target = tokens[target_word]
        cost_one, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        cost += cost_one
        #print(gradIn[source].shape)
        #print(gradPred.shape)
        # +=自更新报错？？？
        gradIn[source] = gradIn[source] + gradPred #输入单个词向量更新
        gradOut += grad

    # print("inputVectors", inputVectors.shape)
    # print("outputVectors", outputVectors.shape)
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # 和skip相反，上下文是输入，中心词是目标
    source = [tokens[source_word] for source_word in contextWords]
    predicted = inputVectors[source]#输入词到词向量映射
    predicted = np.sum(predicted, axis=0)#sum

    target = tokens[currentWord]
    cost_one, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    cost += cost_one
    # +=自更新报错？？？
    for i in source:
        gradIn[i] = gradIn[i] + gradPred #输入上下文词向量更新
    gradOut += grad
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N//2,:]
    outputVectors = wordVectors[N//2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N//2, :] += gin / batchsize / denom
        grad[N//2:, :] += gout / batchsize / denom

    return cost, grad


def t_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print ("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print ("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print ("\n=== Results ===")
    print (skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))
    print (cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))


if __name__ == "__main__":
    # t_normalize_rows()
    t_word2vec()