import numpy as numpy
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm, tqdm_notebook

class KFoldCV:

    def __init__(self, estimator, k_splits, param_grid, cv, scoring=None, iid=False,
                 verbose=False, n_jobs=None, return_train_score=False, random_state=0):
        self.estimator = estimator
        self.k_splits = k_splits
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.iid = iid
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score
        self.random_state = random_state


    def get_score(self, X, y, notebook=False):
        """
        K-fold validation to evaluate the models

        Parameters
        ----------
        X : numpy.array
            Features
        y : numpy.array
            Objective variables
        notebook : bool, optional
            When you use this method on jupyter notebook,
            this should be Ture, by default False
        
        Returns
        -------
        list of float
            The list of scores of each fold 
        
        Attributes
        ----------
        scores : list of float
        best_estimators : list of estimator
        train_indexes : list
        valid_indexes : list
        """
        if notebook:
            pb = tqdm_notebook
        else:
            pb = tqdm

        kf = KFold(n_splits=self.k_splits, random_state=self.random_state) 
        self.scores = []
        self.best_estimators = []
        self.train_indexes = []
        self.valid_indexes = []

        for train_index, valid_index in pb(kf.split(X)):
            self.train_indexes.append(train_index)
            self.valid_indexes.append(valid_index)
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            self.clf = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid,
                                    scoring=self.scoring, n_jobs=self.n_jobs, cv=self.cv,
                                    verbose=self.verbose, return_train_score=self.return_train_score)
            self.clf.fit(X_train, y_train)
            best_estimator = self.clf.best_estimator_ 
            score = best_estimator.score(X_valid, y_valid)
            self.best_estimators.append(best_estimator)
            self.scores.append(score)
        
        return self.scores