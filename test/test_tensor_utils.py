import unittest

import numpy as numpy

from ml.tensor_utils import unique, get_top


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

y = np.arange(len(X))


class TestTensorUtils(unittest.TestCase):

    def test_unique(self):

        
