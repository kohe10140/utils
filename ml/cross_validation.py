import numpy as numpy
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm, tqdm_notebook

class KFoldCV:

    def __init__(self, estimator, k_splits, param_grid, cv, scoring=None, 
                 verbose=False, n_jobs=None, return_train_score=False, random_state=0):
        self.estimator = estimator
        self.k_splits = k_splits
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score
        self.random_state = random_state


    def get_score(self, X, y, notebook=False):
        if notebook:
            pb = tqdm_notebook
        else:
            pd = tqdm

        kf = KFold(n_splits=self.k_splits, random_state=self.random_state) 
        scores = []

        for train_index, valid_index in pd(kf.split(X)):
            X_train, X_valid = X[train_index], y[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            self.clf = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid,
                                    scoring=self.scoring, n_jobs=self.n_jobs, cv=self.cv,
                                    verbose=self.verbose, return_train_score=self.return_train_score)
            self.clf.fit(X_train, y_train)
            best_estimator = self.clf.best_estimator_ 
            score = best_estimator.score(X_valid, y_valid)
            scores.append(score)
        
        return scores
            
            
            
            
            