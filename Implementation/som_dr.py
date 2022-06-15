import numpy as np
from sklearn_som.som import SOM
import cvxpy as cp

from dim_red import DimRed
from counterfactual import CounterfactualExplanation
from utils import non_zero_threshold


default_solver = cp.SCS


class SomDimRed(DimRed):
    def __init__(self, **kwds):
        self.model = None
        self.n_size = None

        super().__init__(**kwds)
    
    def fit(self, X):
        self.n_size = int(np.sqrt(X.shape[0]))
        self.model = SOM(m=self.n_size, n=self.n_size, dim=X.shape[1])
        self.model.fit(X)

    def transform(self, X):
        X_transformed = []

        if len(X.shape) == 1:
            X_idx = self.model.predict(X.reshape(1, -1))
        else:
            X_idx = self.model.predict(X)
        for idx in X_idx:           # Convert prototype indices to two dimensional coordinates
            x_idx = idx % self.n_size
            y_idx = int(idx / self.n_size)
            
            X_transformed.append([x_idx, y_idx])

        return np.array(X_transformed)


class SomDimRedCf(CounterfactualExplanation):
    def __init__(self, model, **kwds):
        if not isinstance(model, SomDimRed):
            raise TypeError(f"'model' must be an instance of 'SomDimRed' not of {type(model)}")

        self.model = model
        self.solver = default_solver
        self.solver_verbosity = False
        self.epsilon = 1e-2

        super().__init__(**kwds)

    def __compute_counterfactual(self, x_orig, y_cf, features_whitelist=None):
        # (Target) prototype
        prototypes = self.model.model.weights
        target_prototype_idx = self.model.n_size * y_cf[1] + y_cf[0]

        # Build and solve linear program for computing a counterfactual explanation
        try:
            dim = x_orig.shape[0] 
            if features_whitelist is None:
                features_whitelist = list(range(dim))

            x = cp.Variable(dim)

            constraints = []    # Requested output
            for idx in range(prototypes.shape[0]):
                if idx == target_prototype_idx:
                    continue

                p_t = prototypes[target_prototype_idx,:]
                p_j = prototypes[idx,:]
                constraints.append(2. * x @ (p_j - p_t) + p_t @ p_t - p_j @ p_j + self.epsilon <= 0)

            A = []  # Some features must not change!
            a = []
            for j in range(dim):
                if j not in features_whitelist:
                    t = np.zeros(dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
            if len(A) != 0:
                A = np.array(A)
                a = np.array(a)
                constraints += [A @ x == a]

            f = cp.Minimize(cp.norm(x_orig - x, 1))
            prob = cp.Problem(f, constraints)

            prob.solve(solver=self.solver, verbose=self.solver_verbosity)

            return x.value
        except Exception as ex:
            print(ex)
            return None

    def _compute_diverse_counterfactual(self, x_orig, y_cf, X_cf):
        features_whitelist = [] # Diversity: Already used features must not be used again
        for x_cf in X_cf:
            delta_cf = np.abs(x_orig - x_cf)
            features_whitelist += [int(idx) for idx in np.argwhere(delta_cf > non_zero_threshold)]
        features_whitelist = list(set(list(range(x_orig.shape[0]))) - set(features_whitelist))

        return self.__compute_counterfactual(x_orig, y_cf, features_whitelist)   

    def compute_explanation(self, x_orig, y_cf):
        return self.__compute_counterfactual(x_orig, y_cf)
