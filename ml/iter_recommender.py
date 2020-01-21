import os

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd


class Iterative:
    
    def __init__(self, estimator, df, y_col):
        """
        Parameters
        ----------
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.

        df : pandas.DataFrame
            The dataframe that has features and target

        y_col : str
            The name of objective variables in df
        """
        self.estimator = estimator
        self.df = df
        self.df_X = df[df.columns[df.columns != y_col]]
        self.df_y = df[y_col]


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


    def get_top_index(self, top_n, df_pred, eval_criteria=False, sym='mean'):
        """
        To get the indexes of data whose top nth predictions 
        
        Parameters
        ----------
        top_n : int
            Top nth score in all the predictions

        df_pred : pandas.DataFrame 
            Predictions in the exploration space 
        
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

        sym : int
           The method to process symmetorical entries, by default 'mean'
           'max', 'min'

        Returns
        -------
        numpy array of shape (n_top) 
            The indexes of data whose top nth predictions
        """
        self.df_pred = df_pred
        self.top_n = top_n
        self.eval_criteria = eval_criteria

        if not eval_criteria:
            eval_criteria = self._intact
        
        if sym == 'mean':
            self.df_pred = self.df_pred.groupby(df_pred.index).mean()
        if sym == 'max':
            self.df_pred = self.df_pred.groupby(df_pred.index).max()
        if sym == 'min':
            self.df_pred = self.df_pred.groupby(df_pred.index).min()
        
        self.top_score_index = eval_criteria(df_pred, top_n)
        return self.top_score_index


    def parse_data(self, exist_index, add_index, exploration_index):
        """
        To add the recommended data to exist_data from exploration_data
        and delete recommended_data from exploration_data

        Parameters
        ----------
        exist_index : array-like
            The index of the train data

        add_index : array-like
            The index of exploration_data to be added to exist_data

        exploration_index : array-like
            The index of the data in exploration space

        Returns
        -------
        update_exist : array-like
            The exist index of added recommendations
                
        update_exploration : array-like
            The exploration index without recommendations

        """
        add_index = add_index.astype(int)
        update_exist = np.unique(np.hstack([exist_index, add_index]))
        update_exploration = np.unique(np.array([key for key in exploration_index if key not in add_index]))
        return update_exist, update_exploration


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
        array-like
            The initial index of training(exist) data and exploration data
        """
        np.random.seed(random_state)
        self.initial_indexes = initial_indexes
        self.n_initial = n_initial

        if initial_indexes == 'random':
            self.initial_indexes = np.random.choice(list(set(self.df_y.index)),
                                                    self.n_initial,
                                                    replace=False)
        else:
            self.initial_indexes = initial_indexes
        return self.initial_indexes


    def _intact(self, df_y, top_n):
        df_rank = df_y.rank(ascending=False)
        index = df_rank[df_rank['y']<=top_n].index
        return index.values 


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
        self.train_index = self.initial_data(n_initial, initial_indexes, random_state)
        self.train_x = self.df_X.loc[self.train_index].values
        self.train_y = self.df_y.loc[self.train_index].values
        self.best_estimator = self.learning(param_grid, x=self.train_x, y=self.train_y)

        rec_output = os.path.join(save_path, 'train'+'0'+'.csv')
        df = self.df.loc[self.train_index]
        df.to_csv(rec_output)

        for i in range(int(len(self.df_y)/top_n)):
            # Recommend iteratively
            print('')
            print(str(i)+'th Recommendation')
            self.exploration_X = self.df_X.drop(self.train_index)
            pred_y = self.best_estimator.predict(self.exploration_X.values)
            df_pred_y = pd.DataFrame(pred_y, index=self.exploration_X.index, columns=['y'])
            self.top_score_index = self.get_top_index(top_n, df_pred_y, eval_criteria)
            self.train_index, self.exploration_index = self.parse_data(self.train_X.index,
                                                                       self.top_score_index,
                                                                       self.exploration_X.index)

            # Save the used estimator and recommended data
            estimators.append(self.best_estimator)
            rec_output = os.path.join(save_path, 'train'+str(i+1)+'.csv')
            df = pd.merge(self.df_X.loc[self.train_index], self.df_y.loc[self.train_index], left_index=True, right_index=True)
            df.to_csv(rec_output)

            self.train_x = self.df_X.loc[self.train_index].values
            self.train_y = self.df_y.loc[self.train_index].values

            if retune:
                self.best_estimator = self.learning(param_grid, X=self.train_X, y=self.train_y)
            
        est_output = os.path.join(save_path, 'estimators.cmp')
        joblib.dump(estimators, est_output, 3)
        return estimators