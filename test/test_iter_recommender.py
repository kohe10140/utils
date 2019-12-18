import unittest

import numpy as np
from sklearn.linear_model import ridge

from ml.iter_recommender import Iterative

X = np.array([[-2, -1, 0],
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

y = np.array([2, 1, 0, -1])

iterative = Iterative(ridge, X, y)


class TestIterative(unittest.TestCase):

    def test_get_index(self):
        top_ns = [2, 4]
        pred = y

        expect_score_index = [np.array([0, 1]),
                              np.array([0, 1, 2, 3])]
        
        for top_n, e_index in zip(top_ns, expect_score_index):
            a_index = iterative.get_top_index(top_n, pred)
            np.testing.assert_array_equal(set(a_index), set(e_index))
    

    def test_parse_data(self):

        exist_data = [np.array([[-2, -1, 0],
                                [1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]),
                      np.array([2, 1, 0, -1])]

        exploration_data = [np.array([[-1, 2, 5],
                                      [3, 2, 3],
                                      [14, -5, 6],
                                      [-7, 8, 29],
                                      [10, 11, 12]]),
                            np.array([2, 1, 0, -1, -2])]

        e_exist_data = [np.array([[-2, -1, 0],
                                  [1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9],
                                  [10, 11, 12]]),
                        np.array([2, 1, 0, -1, -2])]
        
        e_exploration_data = [np.array([[-1, 2, 5],
                                        [3, 2, 3],
                                        [14, -5, 6],
                                        [-7, 8, 29]]),
                              np.array([2, 1, 0, -1])]

        a_exist_data, a_exploration_data = iterative.parse_data(exist_data[0], 4, exploration_data[0], 'v')
        a_exist_data_, a_exploration_data_ = iterative.parse_data(exist_data[1], 4, exploration_data[1], 'h')
        np.testing.assert_array_equal(a_exist_data, e_exist_data[0])
        np.testing.assert_array_equal(a_exploration_data, e_exploration_data[0])
        np.testing.assert_array_equal(a_exist_data_, e_exist_data[1])
        np.testing.assert_array_equal(a_exploration_data_, e_exploration_data[1])

        
        exist_data = [np.array([[-2, -1, 0],
                                [1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]),
                      np.array([2, 1, 0, -1])]

        exploration_data = [np.array([[-1, 2, 5],
                                      [3, 2, 3],
                                      [14, -5, 6],
                                      [-7, 8, 29],
                                      [10, 11, 12]]),
                            np.array([2, 1, 0, -1, -2])]

        e_exist_data = [np.array([[-2, -1, 0],
                                  [1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9],
                                  [14, -5, 6],
                                  [10, 11, 12]]),
                        np.array([2, 1, 0, -1, 0, -2])]
        
        e_exploration_data = [np.array([[-1, 2, 5],
                                        [3, 2, 3],
                                        [-7, 8, 29]]),
                              np.array([2, 1, -1])]

        a_exist_data, a_exploration_data = iterative.parse_data(exist_data[0], [2, 4], exploration_data[0], 'v')
        a_exist_data_, a_exploration_data_ = iterative.parse_data(exist_data[1], [2, 4], exploration_data[1], 'h')
        np.testing.assert_array_equal(a_exist_data, e_exist_data[0])
        np.testing.assert_array_equal(a_exploration_data, e_exploration_data[0])
        np.testing.assert_array_equal(a_exist_data_, e_exist_data[1])
        np.testing.assert_array_equal(a_exploration_data_, e_exploration_data[1])

   
    def test_get_initial(self):
        ret = iterative.initial_data(n_initial=2)
        for i in ret:
            print(i)

if __name__ == "__main__":
    unittest.main()