from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

class Main:
    def __init__(self, data, algo, config_path=False):
        self.data = data
        self.algo = algo
        self.config_path = config_path
        self.scores = []

    def params(self):
        if not self.config_path:
            for k in self.algo.get_params().keys():
                x = input(f'Enter value for {k}: ')
                if x.strip():
                    self.scores.append({k: x})
        return self.scores

    def using_algo(self):
        for param_dict in self.params():
            for key, value in param_dict.items():
                setattr(self.algo, key, value)

        self.algo.fit(self.data)
        score = silhouette_score(self.data, self.algo.labels_)
        print("Silhouette Score:", score)

# Example usage:
X = np.random.rand(10, 20)
finall = Main(X, KMeans())  # Example: Setting a default value for n_clusters
finall.using_algo()

