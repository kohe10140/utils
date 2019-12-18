import numpy as np
from sklearn.model_selection import GridSearchCV


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


    def learning(self, param_grid, X, y, n_jobs=-1, verbose=3):
        self.param_grid = param_grid
        self.gs = GridSearchCV(self.estimator, param_grid, n_jobs=n_jobs, verbose=verbose)
        self.gs.fit(X, y)
        self.learned_model = self.gs.best_estimator_
        return self.learned_model


    def predict(self, pred_X):
        pred_y = self.learned_model.predict(pred_X)
        return pred_X, pred_y


    def get_top_index(self, top_n, pred, eval_criteria=False):
        """
        To get the indexes of data whose top nth predictions 
        
        Parameters
        ----------
        top_n : int
            Top nth score in all the predictions

        pred : array-like of shape (n_prediction)
            Predictions in the exploration space 

        eval_criteria : function, optional
            The criteria to evaluate the predicted data, by default self._intact
        
        Attributes
        ----------
        pred_criteria : numpy array
            The array of transformed scores to evaluate

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
        self.pred_criteria = np.array([eval_criteria(data) for data in pred])

        if top_n < len(self.pred_criteria):
            self.top_score_index = np.argpartition(-self.pred_criteria, top_n)[:top_n]
        else:
            self.top_score_index = np.arange(len(self.pred_criteria))

        return self.top_score_index


    def parse_data(self, exist_data, add_index, exploration_data, stacking):
        """
        To add the recommended data to exist_data from exploration_data
        and delete recommended_data from exploration_data

        Parameters
        ----------
        exist_data : numpy array of shape(n_samples, n_features) or (n_samples)            The data to train.
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
        np.random.random_state(random_state)
        self.initial_indexes = initial_indexes
        self.n_initial = n_initial

        if initial_indexes == 'random':
            initial_indexes = np.random.choice(np.arange(len(self.y)), self.n_initial)

        self.train_X = self.X[initial_indexes]
        self.train_y = self.y[initial_indexes]

        ind = np.ones(len(self.y), dtype=bool)
        ind[initial_indexes] = False
        self.exploration_X = self.X[ind]
        self.exploration_y = self.y[ind]

        return self.train_X, self.train_y, self.exploration_X, self.exploration_y


    def _intact(self, x):
        return x