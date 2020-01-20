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
    r1_list = r1.split(sep=sep)
    r2_list = r2.split(sep=sep)

    if (r1_list[0]==r2_list[1]) & (r1_list[1]==r2_list[0]):
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
    sym = entry1[0]==entry2[1] and \
          entry1[1]==entry2[0] and \
          entry1[2]==entry2[2] and \
          ratio_sym(entry1[3], entry2[3])

    ide = all(entry1==entry2)

    if sym or ide:
        return True
    else:
        return False


def permutate(li):
    """
    Have the list permutated
    Parameters
    ----------
    li : array-like
        The array to be arranged alphabetically
    
    Returns
    -------
    array-like
        The array to be arranged alphabetically

    Example
    -------
    >>> permutate(['B', 'A', 700, '1:2'])
    ['A', 'B', 700, '2:1']
    """
    if li[0] > li[1]:
        esc = li[0]
        li[0] = li[1]
        li[1] = esc
        ratio = li[-1].split(':')
        esc = ratio[0]
        ratio[0] = ratio[1]
        ratio[1] = esc
        li[-1] = ratio[0] + ':' + ratio[1]
    return li


def identical2mean(str_data, y):
    """
    Parameter
    ---------
    str_data : array-like of string (n_samples)
        The entry to be unique

    y : array-like of shape (n_predictions)
        The array of the predictions

    criteria : function
        The criteria that evaluates whether the two entries are identical.

    Return
    ------
    i_mean : numpy array
        pred to be taken mean in the identical data
    """
    # to take mean of identical entries of y
    i_mean_y = np.zeros(len(y))
    for i in range(len(y)):
        index = np.where(str_data==str_data[i])
        print(index)
        i_mean_y[index] = y[index].mean()
    return i_mean_y


def unique(str_data, pred, criteria=check_double):
    """
    Parameter
    ---------
    str_data : array-like of string (n_samples)
        The array to be unique
        ex) ['A_B_500_1:2', 'C_D_700_2:3']

    pred : array-like of shape (n_predictions)
        The array of the predictions

    criteria : function
        The criteria that evaluates whether the two entries are identical.

    Return
    ------
    unique_data : numpy array
    """
    unique_X, index = np.unique(str_data, return_index=True)
    unique_y = pred[index]
    return unique_X, unique_y


def get_top(str_data, uni_X, uni_y, top_n):
    """
    Parameter
    ---------
    str_data : array-like of string (n_samples)
        The string data like below
        ex) ['A_B_500_1:2', 'C_D_700_2:3']

    uni_X : array-like of string (n_samples(unique))
        The unique array 
    
    uni_y : array-like of shape (n_predictions)
        The unique array of the predictions

    top_n : int
        Top nth score in all the predictions

    Return
    ------
    numpy array
        The top nth index considered symmetory,
        so the size of return array will be larger than n.
    """
    if top_n < len(uni_X):
        index = np.argpartition(-uni_y, top_n)[:top_n]
    else:
        index = np.arange(len(uni_X))
    top_entries = uni_X[index]
    top_n_index = np.array([])
    for entry in top_entries:
        top_n_index = np.hstack([top_n_index, np.where(str_data==entry)[0]])

    return top_n_index


def sym_mean(y, top_n, X):
    str_data = np.array(['_'.join(permutate(entry).astype(str)) for entry in X])
    i_mean_y = identical2mean(str_data, y)
    unique_X, unique_y = unique(str_data, i_mean_y)
    top_n_index = get_top(str_data, unique_X, unique_y, top_n)
    return top_n_index