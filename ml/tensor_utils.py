import numpy as np


def ratio_sym(r1, r2, sep=':'):
    """
    Parameter
    ---------
    r1, r2 : str
        The ratio to be checked

    sep : str
        The string to separate the ratioes, by default ':'

    Return
    ------
    When the inverse of r1 is r2, return True

    Example
    -------
    >>> ratio_sym('2:1', '1:2')
    True
    """
    r1_set = set(r1.split(sep=sep))
    r2_set = set(r2.split(sep=sep))

    if list(r1_set) == list(r2_set):
        return True
    else:
        return False


def check_double(entry1, entry2):
    """
    Parameter
    ---------
    entry1, entry2 : array-like of shape(4)
        The entry to be compared

    Return
    ------
    When the entries is same, return True

    Example
    -------
    >>> entry1 = ['A', 'B', k , '1:2']
    >>> entry2 = ['B', 'A', k , '2:1']
    >>> check_double(entry1, entry2)
    True
    """
    sym = (entry1[0]==entry2[1]) and \
          (entry1[1]==entry2[0]) and \
          (entry1[2]==entry2[2]) and \
          ratio_sym(entry1[-1], entry2[-1])

    ide = entry1 == entry2

    if sym or ide:
        return True
    else:
        return False


def unique(data, pred, criteria=check_double):
    """
    Parameter
    ---------
    data : array-like (n_samples, n_features)
        The array to be unique

    pred : array-like of shape (n_predictions)
        The array of the predictions

    criteria : function
        The criteria that evaluates whether the two entries are identical.

    Return
    ------
    unique_data : numpy array
    """
    unique_X = data[0]
    unipue_y = pred
    for i in range(len(data)):
        for j in unique_X:
            if check_double(data[i], j):
                break
        else:
            np.vstack([unique_X, data[i])
            np.hstack([unique_y, pred[i])
    return unique_X, unique_y


def get_top(X, y, top_n):
    """
    Parameter
    ---------
    X : array-like of shape (n_samples, n_features)
        The unique array 

    y : array-like of shape (n_predictions)
        The unique array of the predictions

    top_n : int
        Top nth score in all the predictions
    Return
    ------
    numpy array
        The top nth index considered symmetory

    """
    data = np.hstack([X, y[:, np.newaxis]])

    if top_n < len(X):
        index = np.argpartition(-y, top_n)[:top_n]
    else:
        index = np.arange(len(X))

    top_n_index = []
    for entry in X[index]:
        for i in range(len(X)):
            if check_double(entry, i):
                top_n_index.append(i)

    return np.array(top_n_index)
