from sklearn.cluster import KMeans
import numpy as np
X = np.random.rand(10, 20)
class Main:
    def __init__(self, data, algo, config_path=False):
        self.data = data
        self.algo = algo
        self.config_path = config_path
        self.scores = []

    def params(self, ):
        if self.config_path == False:
            for k in self.algo.get_params():
                x = input(f'Enter values of {self.algo}, {k} = ')
                # if its empty dont add
                self.scores.append({k:x})
            return self.scores

    def using_algo(self):
        for param_dict in self.params():
            for key, value in param_dict.items():
                return setattr(self.algo, key, value)



finall = Main(X, KMeans())
print(finall.params())
finall.using_algo()

