import unittest
import sys

from sklearn.linear_model import Ridge 
from sklearn.datasets import load_boston
import pandas as pd

from ml.cross_validation import KFoldCV
#class TestKFoldCV(unittest.TestCase):

if __name__ == "__main__":
    
    sys.path.append('/home/nishi/research/utils')

    boston = load_boston() 
    boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
    boston_df['MEDV'] = boston.target 

    rg = Ridge(solver='auto')
    X = boston_df[['RM']].values
    Y = boston_df['MEDV'].values

    param_grid = {'alpha': [0.01, 0.1]}

    kfcv = KFoldCV(rg, 10, param_grid, 3, n_jobs=-1)

    print(kfcv.get_score(X, Y))
