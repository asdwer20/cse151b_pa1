from yaml import load
from data import load_data, z_score_normalize
import numpy as np

if __name__ == "__main__":
    data, labels = load_data()
    # c = 0
    A = np.ones((4,5))
    x = np.arange(5)
    X = np.tile(x,(10,1))

    X_norm = z_score_normalize(X)