# centered kernel alignment
# CVU 2019

import hoggorm as ho
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

def center_kernel(K, copy=True):
    '''
    Centered version of a kernel matrix (corresponding to centering the)
    implicit feature map.
    '''
    means = K.mean(axis=0)
    if copy:
        K = K - means[None, :]
    else:
        K -= means[None, :]
    K -= means[:, None]
    K += means.mean()
    return K

def align(X, Y):
    '''
    Applies pairwise linear kernel to X and Y
    to generate K1 and K2.

    Centers each of these kernels.

    Returns the kernel alignment
        <K1, K2>_F / (||K1||_F ||K2||_F)
    defined by
        Cristianini, Shawe-Taylor, Elisseeff, and Kandola (2001).
        On Kernel-Target Alignment. NIPS.
    Note that the centered kernel alignment of
        Cortes, Mohri, and Rostamizadeh (2012).
        Algorithms for Learning Kernels Based on Centered Alignment. JMLR 13.
    is just this applied to center_kernel()s.
    '''
    K1 = center_kernel(linear_kernel(X, X))
    K2 = center_kernel(inear_kernel(Y, Y))
    return np.sum(K1 * K2) / np.linalg.norm(K1) / np.linalg.norm(K2)

def cka(dataList1, dataList2):
    """
    This function computes the RV matrix correlation coefficients between pairs
    of arrays. The number and order of objects (rows) for the two arrays must
    match. The number of variables in each array may vary.
    Reference: `The STATIS method`_
    .. _The STATIS method: https://www.utdallas.edu/~herve/Abdi-Statis2007-pretty.pdf
    PARAMETERS
    ----------
    dataList : list
    A list holding numpy arrays for which the RV coefficient will be computed.
    RETURNS
    -------
    numpy array
    A numpy array holding RV coefficients for pairs of numpy arrays. The
    diagonal in the result array holds ones, since RV is computed on
    identical arrays, i.e. first array in ``dataList`` against frist array
    in
    Examples
    --------
    >>> import hoggorm as ho
    >>> import numpy as np
    >>>
    >>> # Generate some random data. Note that number of rows must match across arrays
    >>> arr1 = np.random.rand(50, 100)
    >>> arr2 = np.random.rand(50, 20)
    >>> arr3 = np.random.rand(50, 500)
    >>>
    >>> # Center the data before computation of RV coefficients
    >>> arr1_cent = arr1 - np.mean(arr1, axis=0)
    >>> arr2_cent = arr2 - np.mean(arr2, axis=0)
    >>> arr3_cent = arr3 - np.mean(arr3, axis=0)
    >>>
    >>> # Compute RV matrix correlation coefficients on mean centered data
    >>> rv_results = ho.RVcoeff([arr1_cent, arr2_cent, arr3_cent])
    >>> array([[ 1.        ,  0.41751839,  0.77769025],
    [ 0.41751839,  1.        ,  0.51194496],
    [ 0.77769025,  0.51194496,  1.        ]])
    >>>
    >>> # Get RV for arr1_cent and arr2_cent
    >>> rv_results[0, 1]
    0.41751838661314689
    >>>
    >>> # or
    >>> rv_results[1, 0]
    0.41751838661314689
    >>>
    >>> # Get RV for arr2_cent and arr3_cent
    >>> rv_results[1, 2]
    0.51194496245209853
    >>>
    >>> # or
    >>> rv_results[2, 1]
    0.51194496245209853
    """
    # First compute the scalar product matrices for each data set X
    scalArrList = []
    for arr in dataList:
        # center the data
        arr -= np.mean(arr, axis=0)
        scalArr = np.dot(arr, np.transpose(arr))
        scalArrList.append(scalArr)
    # Now compute the 'between study cosine matrix' C
    C = np.zeros((len(dataList), len(dataList)), float)
    for index, element in np.ndenumerate(C):
        nom = np.trace(np.dot(np.transpose(scalArrList[index[0]]),
                            scalArrList[index[1]]))
        denom1 = np.trace(np.dot(np.transpose(scalArrList[index[0]]),
                               scalArrList[index[0]]))
        denom2 = np.trace(np.dot(np.transpose(scalArrList[index[1]]),
                               scalArrList[index[1]]))
        Rv = nom / np.sqrt(np.dot(denom1, denom2))
        C[index[0], index[1]] = Rv
    return C
