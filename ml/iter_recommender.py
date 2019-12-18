import numpy as np


class Iterative:
    
    def __init__(self, algo, X, y):
        self.algo = algo
        self.X = X
        self.y = y


    def learning(self):
        return learned_model


    def predict(self):
        return prediction


    def get_top_index(self, top_n, pred, eval_criteria=self._intact):
        self.pred = pred
        self.top_n = top_n
        self.eval_criteria = eval_criteria
        self.pred_criteria = np.array([eval_criteria(data) for data in pred])
        self.top_score_index = np.argpartition(-self.pred_criteria, top_n)[:top_n]
        return self.top_score_index


    def parse_data(self, exist_data, add_index, exploration_data, stacking):
        if stacking == 'v':
            updated_data = np.vstack([exist_data, exploration_data[add_index]]) 
            updated_exploration = np.delete(exploration_data, add_index, axis=0)
        elif stacking == 'h':
            updated_data = np.hstack([exist_data, exploration_data[add_index]]) 
            updated_exploration = np.delete(exploration_data, add_index, axis=0)
        else:
            raise ValueError
         
        return updatad_exist, updated_exploration


    def initial_data(self, n_initial, initial_index='random'):
        self.initial_index = initial_index
        self.n_initial = n_initial

        if initial_indexes = 'random':
            initial_indexes = np.random.choice(np.arange(len(self.y)), self.n_initial)

        self.train_X = X[initial_indexes]
        self.train_y = X[initial_indexes]

        ind = np.ones(len(self.y), dtype=bool)
        ind[initial_indexes] = False
        self.exploration_X = X[ind]
        self.exploration_y = X[ind]

        return self.train_X, self.train_y, self.test_X, self.test_y


    def _intact(self, x):
        return x