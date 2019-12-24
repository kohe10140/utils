import unittest

import numpy as np

from ml.tensor_utils import identical2mean, unique, get_top, sym_mean, permutate


X = np.array([['A', 'B', 500, '1:2'], #0
              ['B', 'A', 500, '2:1'], #0
              ['A', 'B', 500, '1:2'], #0
              ['A', 'B', 500, '1:1'], #1
              ['B', 'A', 500, '1:1'], #1
              ['A', 'B', 300, '1:2'], #2
              ['A', 'C', 500, '1:2'], #3
              ['C', 'A', 500, '2:1'], #3
              ['A', 'C', 500, '1:2'], #3
              ['A', 'C', 500, '1:1'], #4
              ['C', 'A', 500, '1:1'], #4
              ['A', 'C', 300, '1:2']])#5

y = np.arange(len(X)).astype(float)
y_ = np.arange(len(X)).astype(float)
y__ = np.arange(len(X)).astype(float)
str_data = np.array(['_'.join(permutate(entry).astype(str)) for entry in X])


class TestTensorUtils(unittest.TestCase):

    def test_identical2mean(self):
        str_data = np.array(['_'.join(permutate(entry).astype(str)) for entry in X])
        a_i_mean = identical2mean(str_data=str_data, y=y)
        e_i_mean = np.array([1.0, 1.0, 1.0, 
                             3.5, 3.5, 5.0,
                             7.0, 7.0, 7.0,
                             9.5, 9.5, 11.0])
        np.testing.assert_array_equal(a_i_mean, e_i_mean)


    def test_unique(self):
        str_data = np.array(['_'.join(permutate(entry).astype(str)) for entry in X])
        a_unique_X, a_unique_y = unique(str_data=str_data, pred=y_)
        e_unique_X = np.array(['A_B_300_1:2', 
                               'A_B_500_1:1', 
                               'A_B_500_1:2', 
                               'A_C_300_1:2', 
                               'A_C_500_1:1', 
                               'A_C_500_1:2'])
        e_unique_y = np.array([5, 3, 0, 11, 9, 6])

        np.testing.assert_array_equal(a_unique_X, e_unique_X)
        np.testing.assert_array_equal(a_unique_y, e_unique_y)


    def test_get_top(self):
        unique_X, unique_y = unique(str_data=str_data, pred=y_)
        a_top = get_top(str_data, unique_X, unique_y, 3)
        e_top = np.array([11, 6, 7, 8, 9, 10])
        np.testing.assert_array_equal(a_top, e_top)
        

    def test_sym_mean(self):
        i_mean_y__ = identical2mean(str_data, y__)
        unique_X, unique_y = unique(str_data=str_data ,pred=i_mean_y__)
        a_top = get_top(str_data, unique_X, unique_y, 3)
        e_top = sym_mean(y=y__, top_n=3, X=X)
        np.testing.assert_array_equal(a_top, e_top)


if __name__ == '__main__':
    unittest.main()
        
