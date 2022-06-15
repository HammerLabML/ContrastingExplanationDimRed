import numpy as np
import jax
import jax.numpy as npx
from scipy.optimize import minimize

from jax_stuff import jax_dtype
from counterfactual import CounterfactualExplanation
from autoencoder import AutoencoderDimRed
from utils import non_zero_threshold


class AutoEncoderDimRedCf(CounterfactualExplanation):
    def __init__(self, model, C_reg=.1, solver="CG", max_iter=None, **kwds):
        if not isinstance(model, AutoencoderDimRed):
            raise TypeError(f"'model' must be an instance of 'AutoencoderDimRed' not of {type(model)}")

        self.model = model
        self.C_reg = C_reg
        self.solver = solver
        self.max_iter = max_iter

        super().__init__(**kwds)

    def __compute_counterfactual(self, x_orig, y_cf, features_blacklist=[]):
        A = np.diag(np.ones(x_orig.shape[0]))
        b = np.zeros(x_orig.shape[0])
        for idx in features_blacklist:
            A[idx, idx] = 0.
            b[idx] = x_orig[idx]

        # JAX based loss function
        loss = lambda x: npx.linalg.norm(self.model.transform(A @ x + b) - y_cf, 2) + self.C_reg * npx.linalg.norm(x_orig - x, 1) 

        # Compute gradients via autodiff
        fgrad = jax.grad(loss)
        grad = lambda x: fgrad(npx.array(x, dtype=jax_dtype))

        # Perform gradient based optimization
        res = minimize(fun=loss, x0=x_orig, jac=grad, method=self.solver, options={'maxiter': self.max_iter})
        x_cf = res["x"]

        return x_cf

    def _compute_diverse_counterfactual(self, x_orig, y_cf, X_cf):
        features_blacklist = [] # Diversity: Already used features must not be used again
        for x_cf in X_cf:
            delta_cf = np.abs(x_orig - x_cf)
            features_blacklist += [int(idx) for idx in np.argwhere(delta_cf > non_zero_threshold)]
        features_blacklist = list(set(features_blacklist))

        return self.__compute_counterfactual(x_orig, y_cf, features_blacklist)

    def compute_explanation(self, x_orig, y_cf):
        return self.__compute_counterfactual(x_orig, y_cf)
