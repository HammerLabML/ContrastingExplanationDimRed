import numpy as np
import jax
import jax.numpy as npx
from jax.nn import sigmoid
from scipy.optimize import minimize
from parametric_tSNE import Parametric_tSNE     # https://github.com/jsilter/parametric_tsne


from dim_red import DimRed
from counterfactual import CounterfactualExplanation
from utils import non_zero_threshold
from jax_stuff import jax_dtype


class TsneDimRed(DimRed):
    def __init__(self, **kwds):
        self.model = None

        super().__init__(**kwds)
    
    def fit(self, X):
        self.model = Parametric_tSNE(num_inputs=X.shape[1], num_outputs=2, perplexities=30)
        self.model.fit(X)
    
    def transform(self, X):
        if len(X.shape) == 1:
            return self.model.transform(X.reshape(1, -1))
        else:
            return self.model.transform(X)


class JaxMlp():
    def __init__(self, layers, **kwds):
        self.layers = layers

        super().__init__(**kwds)

    def __call__(self, X):
        x = X
        for lyr in self.layers[:len(self.layers)-1]:
            x = sigmoid(npx.dot(x, lyr[0]) + lyr[1])
        x = npx.dot(x, self.layers[-1][0]) + self.layers[-1][1]
        return x


class TsneDimRedCf(CounterfactualExplanation):
    def __init__(self, model, C_reg=.1, solver="CG", max_iter=None, **kwds):
        self.model = model
        self.C_reg = C_reg
        self.solver = solver
        self.max_iter = max_iter

        # Rebuild model using jax
        layers = []
        for layer in self.model.model.model.layers:
            layers.append(layer.get_weights())
        self.jax_model = JaxMlp(layers)

        super().__init__(**kwds)
    
    def __compute_counterfactual(self, x_orig, y_cf, features_blacklist=[]):
        A = np.diag(np.ones(x_orig.shape[0]))
        b = np.zeros(x_orig.shape[0])
        for idx in features_blacklist:
            A[idx, idx] = 0.
            b[idx] = x_orig[idx]

        # JAX based loss function
        loss = lambda x: npx.linalg.norm(self.jax_model(A @ x + b) - y_cf, 2) + self.C_reg * npx.linalg.norm(x_orig - x, 1) 

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