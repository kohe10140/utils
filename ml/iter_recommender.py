import os

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd


class Iterative:
    
    def __init__(self, estimator, X, y):
        """
        Parameters
        ----------
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.

        X : array-like of shape (n_samples, n_features)
            Features

        y : array like of shape (n_samples)
            Objective variables
        """
        self.estimator = estimator
        self.X = X
        self.y = y


    def learning(self, param_grid, X, y, n_jobs=-1, verbose=3, cv=5):
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.cv= cv
        self.gs = GridSearchCV(self.estimator, param_grid, n_jobs=n_jobs, verbose=verbose)
        self.gs.fit(X, y)
        self.learned_model = self.gs.best_estimator_
        return self.learned_model


    def predict(self, pred_X):
        pred_y = self.learned_model.predict(pred_X)
        return pred_X, pred_y


    def get_top_index(self, top_n, pred, exploration_X, eval_criteria=False):
        """
        To get the indexes of data whose top nth predictions 
        
        Parameters
        ----------
        top_n : int
            Top nth score in all the predictions

        pred : array-like of shape (n_prediction)
            Predictions in the exploration space 
        
        exploration_X : array-like of shape (n_samples, n_features)
            features in the exploration space

        eval_criteria : function, optional
            The criteria to evaluate the predicted data, by default self._intact
            The argument of this function must be (y, top_n, X)
            Parameters
            ----------
            y : array-like of shape (n_predictions)
                The array of predictions

            top_n : int
                
            X : array-like of shape (n_samples, n_features)
                The array of features

            Returns
            -------
            top_score_index : numpy array
                The index of top nth score 

        Returns
        -------
        numpy array of shape (n_top) 
            The indexes of data whose top nth predictions
        """
        self.pred = pred
        self.top_n = top_n
        self.eval_criteria = eval_criteria

        if not eval_criteria:
            eval_criteria = self._intact
        
        self.top_score_index = eval_criteria(pred, top_n, exploration_X)
        return self.top_score_index


    def parse_data(self, exist_data, add_index, exploration_data, stacking):
        """
        To add the recommended data to exist_data from exploration_data
        and delete recommended_data from exploration_data

        Parameters
        ----------
        exist_data : numpy array of shape(n_samples, n_features) or (n_samples)  
            The data to train

        add_index : int or numpy array
            The index of exploration_data to be added to exist_data

        exploration_data : numpy array of shape(n_samples, n_features) or (n_samples)
            The data in exploration space

        stacking : str
            'v' -> Stacking vertically 
            's' -> Stacking horizontally
        
        Returns
        -------
        update_exist : numpy array
            The exist data added recommendations
                
        update_exploration : numpy array
            The exploration data without recommendations

        Raises
        ------
        ValueError
        """
        add_index = add_index.astype(int)
        if stacking == 'v':
            updated_exist = np.vstack([exist_data, exploration_data[add_index]]) 
            updated_exploration = np.delete(exploration_data, add_index, axis=0)
        elif stacking == 'h':
            updated_exist = np.hstack([exist_data, exploration_data[add_index]]) 
            updated_exploration = np.delete(exploration_data, add_index, axis=0)
        else:
            raise ValueError
         
        return updated_exist, updated_exploration


    def initial_data(self, n_initial, initial_indexes='random', random_state=0):
        """
        
        Parameters
        ----------
        n_initial : int
            The number of initial data for iterative recommendation
        initial_index : str, optional
            The method to select initial data, by default 'random'
        
        Returns
        -------
        The initial data of training(exist) data and exploration data
        """
        np.random.seed(random_state)
        self.initial_indexes = initial_indexes
        self.n_initial = n_initial

        if initial_indexes == 'random':
            self.initial_indexes = np.random.choice(np.arange(len(self.y)),
                                                    self.n_initial,
                                                    replace=False)
        elif initial_indexes == 'sym_random':
            self.initial_indexes = np.random.choice(np.arange(int(len(self.y)/2)),
                                                    self.n_initial,
                                                    replace=False)
            sym_indexes = self.initial_indexes + np.full(len(self.initial_indexes), int(len(self.y)/2))
            self.initial_indexes = np.hstack([self.initial_indexes, sym_indexes]).astype(int)
        else:
            self.initial_indexes = initial_indexes

        self.train_X = self.X[self.initial_indexes]
        self.train_y = self.y[self.initial_indexes]

        self.exploration_X = np.delete(self.X, self.initial_indexes, axis=0)#self.X[ind]
        self.exploration_y = np.delete(self.y, self.initial_indexes, axis=0)#self.y[ind]

        return self.train_X, self.train_y, self.exploration_X, self.exploration_y


    def _intact(self, y, top_n, X):

        if top_n < len(y):
            index = np.argpartition(-y, top_n)[:top_n]
        else:
            index = np.arange(len(y))
        return index 


## EXAMPLE OF ITERATIVE RECOMENDATION ####################
    def iter_recommend(self, n_initial, random_state,
                       param_grid, save_path, top_n,
                       initial_indexes='random', n_jobs=-1,
                       eval_criteria=False, retune=False):
        """
        Recommend and output the result

        Parameters
        ----------
        save_path : str
            The parent directory path of output files
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        estimators = []

        # Extract initial data
        self.train_X, self.train_y, self.exploration_X, self.exploration_y = self.initial_data(n_initial, initial_indexes, random_state)
        self.best_estimator = self.learning(param_grid, X=self.train_X, y=self.train_y)

        rec_output = os.path.join(save_path, 'train'+'0'+'.cmp')
        df = pd.DataFrame(self.train_X)
        df['y'] = self.train_y
        df.to_csv(rec_output)

        for i in range(int(len(self.y)/top_n)):
            # Recommend iteratively
            print('')
            print(str(i)+'th Recommendation')
            pred_y = self.best_estimator.predict(self.exploration_X)
            self.top_score_index = self.get_top_index(top_n, pred_y, self.exploration_X, eval_criteria)
            self.train_X, self.exploration_X = self.parse_data(self.train_X, self.top_score_index, self.exploration_X, stacking='v')
            self.train_y, self.exploration_y = self.parse_data(self.train_y, self.top_score_index, self.exploration_y, stacking='h')

            # Save the used estimator and recommended data
            estimators.append(self.best_estimator)
            rec_output = os.path.join(save_path, 'train'+str(i+1)+'.cmp')
            df = pd.DataFrame(self.train_X)
            df['y'] = self.train_y
            df.to_csv(rec_output)

            if retune:
                self.best_estimator = self.learning(param_grid, X=self.train_X, y=self.train_y)
            
        est_output = os.path.join(save_path, 'estimators.cmp')
        joblib.dump(estimators, est_output, 3)
        return estimators