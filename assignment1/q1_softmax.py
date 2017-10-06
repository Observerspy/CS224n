import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        x_max = np.max(x, axis=1).reshape(x.shape[0], 1)
        x -= x_max
        exp_sum = np.sum(np.exp(x), axis=1).reshape(x.shape[0], 1)
        x = np.exp(x) / exp_sum
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x_max = np.max(x)
        x -= x_max
        exp_sum = np.sum(np.exp(x))
        x = np.exp(x) / exp_sum
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print ("test1", test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print ("test2", test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print ("test3", test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print ("You should be able to verify these results by hand!\n")


def t_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print ("Running your tests...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    B = np.array([[1,2,3]])
    print(A[0,:].shape)
    print(B.shape)
    # C = A[2, :] - B
    A[2] = A[2] - B
    # A[2] -= B
    print(A)
    softmax_basic()
    #test_softmax()
