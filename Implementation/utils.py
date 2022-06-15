import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes, load_digits


non_zero_threshold = 1e-5


def scale_standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def create_toy_data(n_samples=500, n_dim=10):
    X = []
    y = []

    # Random centers [0,100) of Gaussian distributions
    centers = 100. * np.random.random(n_dim)

    # Generate data by sampling from a different Gaussian distribution for every dimension
    for mu in centers:
        X.append(np.random.normal(mu, size=n_samples))
    X = np.array(X).T

    # Cluster data into two groups -- labeling according to cluster assignment
    model = KMeans(n_clusters=2)
    model.fit(X)
    y = model.predict(X)

    return X, y

def load_data(data_desc, scaling=True):
    if data_desc == "iris":
        X, y = load_iris(return_X_y=True)
    elif data_desc == "toy":
        X, y = create_toy_data()
    elif data_desc == "breastcancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif data_desc == "wine":
        X, y = load_wine(return_X_y=True)
    elif data_desc == "boston":
        X, y = load_boston(return_X_y=True)
    elif data_desc == "diabetes":
        X, y = load_diabetes(return_X_y=True)
    elif data_desc == "digits":
        X, y = load_digits(return_X_y=True)
    else:
        raise ValueError(f"Unknown data set '{data_desc}'")

    if scaling is True:
        X = scale_standardize_data(X)
    return X, y