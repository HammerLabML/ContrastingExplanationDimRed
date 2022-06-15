import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from autoencoder import AutoencoderModel, AutoencoderDimRed
from utils import load_data, non_zero_threshold
from linear_dr import LinearDimRed, LinearDimRedCf
from ae_dr import AutoEncoderDimRedCf
from som_dr import SomDimRedCf, SomDimRed
from tsne_dr import TsneDimRed, TsneDimRedCf
from memory_counterfactual import MemoryCounterfactual


def shift_perturbation(X, perturbed_features, magnitude=5.):
    X = np.copy(X)
    for j in perturbed_features:
        X[:, j] += magnitude * X[:,j].std()

    return X

def gaussian_perturbation(X, perturbed_features, variance=5.):
    X = np.copy(X)
    for j in perturbed_features:
        X[:, j] += np.random.normal( 0 , variance * X[:,j].std(), size=X.shape[0])
    
    return X

def zero_perturbation(X, perturbed_features):
    for j in perturbed_features:
        X[:,j] = 0
    
    return X

def permutation_perturbation(X, perturbed_features):
    X_results = np.copy(X)
    for j in perturbed_features:
        perm = np.random.permutation(X.shape[0])
        X_results[:, j] = X[:, j][perm]

    return X_results


def apply_perturbation(X, perturbation_desc, perturbed_features, y):
    if perturbation_desc == "shift":
        return shift_perturbation(X, perturbed_features)
    elif perturbation_desc == "gaussian":
        return gaussian_perturbation(X, perturbed_features)
    elif perturbation_desc == "zero":
        return zero_perturbation(X, perturbed_features)
    elif perturbation_desc == "permute":
        return permutation_perturbation(X, perturbed_features)
    elif perturbation_desc == "none":
        # Randomly select a sample with a different label
        X_result = np.zeros(X.shape)
        for idx in range(X.shape[0]):
            y_orig = y[idx]
            candiate_indices = np.where(y != y_orig)[0]
            
            target_idx = random.choice(candiate_indices)
            X_result[idx,:] = X[target_idx,:]

        return X_result
    else:
        raise ValueError(f"Unkown perturbation '{perturbation_desc}'")


def evaluate_diversity(explanations):
    r = 0.

    for i in range(len(explanations)):
        for j in range(i+1, len(explanations)):
            expl1 = explanations[i]
            expl2 = explanations[j]

            r += np.sum([1. if expl1[k] > non_zero_threshold and expl2[k] > non_zero_threshold else 0. for k in range(len(expl1))])
    
    return r

def evaluate_results(explanations, ground_truth_relevant_features, transformed_samples_dist, cf_transformed_sanmples_error):
    results = []
    
    for i in range(len(explanations)):
        Expl = explanations[i]  # List of explanations!
        transformed_sample_dist = transformed_samples_dist[i]
        cf_transformed_error = np.mean(cf_transformed_sanmples_error[i])

        # Collect relevant features from all explanations
        indices = []
        for expl in Expl:
            indices += [int(idx) for idx in np.argwhere(expl > non_zero_threshold)]
        indices = list(set(indices))
        expl_shape = Expl[0].shape[0]

        #sparsity = len(indices) / expl_shape   # Sparsity of merged explanations
        sparsity = np.mean([len([int(idx) for idx in np.argwhere(expl > non_zero_threshold)]) / expl_shape for expl in Expl])  # Mean sparsity over all explanations

        # Diversity
        diversity = evaluate_diversity(Expl)

        # Compute confusion matrix
        tp = np.sum([idx in ground_truth_relevant_features for idx in indices]) / len(indices)
        fp = np.sum([idx not in ground_truth_relevant_features for idx in indices]) / len(indices)
        tn = np.sum([idx not in ground_truth_relevant_features for idx in filter(lambda i: i not in indices, range(expl_shape))]) / len(list(filter(lambda i: i not in indices, range(expl_shape))))
        fn = np.sum([idx in ground_truth_relevant_features for idx in filter(lambda i: i not in indices, range(expl_shape))]) / len(list(filter(lambda i: i not in indices, range(expl_shape))))

        # Compute recall and precision
        tp_ = np.sum([idx in ground_truth_relevant_features for idx in indices])
        fp_ = np.sum([idx not in ground_truth_relevant_features for idx in indices])
        tn_ = np.sum([idx not in ground_truth_relevant_features for idx in filter(lambda i: i not in indices, range(expl_shape))])
        fn_ = np.sum([idx in ground_truth_relevant_features for idx in filter(lambda i: i not in indices, range(expl_shape))])


        recall = tp_ / (tp_ + fn_)
        precision = tp_ / (tp_ + fp_)
        f1 = 2. * tp_ / (2. * tp_ + fp_ + fn_)

        results.append({"diversity": diversity, "sparsity": sparsity,"recall": recall, "precision": precision, "f1": f1, "tp": tp_, "fp": fp_, "tn": tn_, "fn": fn_, "tpr": tp, "fpr": fp, "tnr": tn, "fnr": fn, "transformed_dist": transformed_sample_dist, "cf_error": cf_transformed_error})

    return pd.DataFrame(results)

def summarize_evaluation(eval_results):
    return eval_results.mean(), eval_results.var()


def plot_data(X, y, show_legend=True, xticks=[], yticks=[], title="", show=True, savefig_path=None):
    plt.figure()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    if X.shape[1] == 2:
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)#, c=list(map(lambda i: colors[i], y)), cmap=cmap_bold)

        plt.ylim(y_min, y_max)

    plt.xlim(x_min, x_max)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.title(title)
    
    if show:
        if savefig_path is None:
            plt.show()
        else:
            plt.savefig(savefig_path)


n_iter = 100


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: <Description of data set> <Description of perturbation> <n_perturbed_features> <Desciption of dimensionality reduction method> <Raw results output directory> <Computation method>")
        os._exit(1)
    
    dataset_desc = sys.argv[1]
    perturbation_desc = sys.argv[2]
    n_perturbed_features = int(sys.argv[3])
    model_desc = sys.argv[4]
    raw_results_out = sys.argv[5]
    computation_method = sys.argv[6]

    n_explanations = 3

    print(dataset_desc,perturbation_desc,n_perturbed_features,model_desc,raw_results_out,computation_method)

    def run_exp(dataset_desc, perturbation_desc, model_desc):
        # Load data
        X, y = load_data(dataset_desc)
        n_dim = X.shape[1]
        
        # Fit and apply dimensionality reduction method
        if model_desc == "linear":
            model = LinearDimRed()
        elif model_desc == "ae":
            model = AutoencoderDimRed(AutoencoderModel(features=[10, 2], input_dim=X.shape[1]))
        elif model_desc == "som":
            model = SomDimRed()
        elif model_desc == "tsne":
            model = TsneDimRed()
        else:
            raise ValueError("Unknown model")

        X_red = model.fit_transform(X)

        # Apply perturbation to data
        perturbed_features = random.choices(list(range(n_dim)), k=n_perturbed_features)
        X_perturbed = apply_perturbation(X, perturbation_desc, perturbed_features, y)
        X_perturbed_red = model.transform(X_perturbed)

        # Visualize data
        #plot_data(X_red, y)
        #plot_data(X_perturbed_red, y)
        #os._exit(0)

        # Compute counterfactual explanation of dimensionality reduction
        if computation_method == "fancy":
            if model_desc == "linear":
                explainer = LinearDimRedCf(model, C_pred=10.)
            elif model_desc == "ae":
                explainer = AutoEncoderDimRedCf(model)
            elif model_desc == "som":
                explainer = SomDimRedCf(model)
            elif model_desc == "tsne":
                explainer = TsneDimRedCf(model)
        else:
            explainer = MemoryCounterfactual(X, X_red)

        explanations = []
        transformed_samples_dist = []
        cf_transformed_sanmples_error = []
        raw_results = {"X_orig": [], "X_transformed": [], "X_cf": [], "Y_cf": [], "X_cf_transformed": []}
        for i in range(X.shape[0]):
            x_orig = X[i,:]
            y_orig = X_red[i,:]
            y_cf = X_perturbed_red[i,:]
        
            transformed_dist = np.linalg.norm(y_orig - y_cf, 2)

            X_cf_ = explainer.compute_diverse_explanations(x_orig, y_cf, n_explanations=n_explanations)
            if X_cf_ is None:
                continue
            Y_cf_pred = [model.transform(x_cf) for x_cf in X_cf_]

            cf_transformed_error = [np.linalg.norm(y_cf - y_cf_pred, 2) for y_cf_pred in Y_cf_pred]

            Delta_cf = [np.abs(x_orig - x_cf) for x_cf in X_cf_]

            explanations.append(Delta_cf)
            transformed_samples_dist.append(transformed_dist)
            cf_transformed_sanmples_error.append(cf_transformed_error)

            raw_results["X_orig"].append(x_orig)
            raw_results["X_transformed"].append(y_orig)
            raw_results["Y_cf"].append(y_cf)
            raw_results["X_cf"].append(X_cf_)
            raw_results["X_cf_transformed"].append(Y_cf_pred)

        #print(explanations)
        return evaluate_results(explanations, perturbed_features, transformed_samples_dist, cf_transformed_sanmples_error), raw_results
    
    results = Parallel(n_jobs=-2)(delayed(run_exp)(dataset_desc, perturbation_desc, model_desc) for _ in range(n_iter))
    results_stat = [r for r, _ in results]
    raw_results = [r for _, r in results]
    results_stat = pd.concat(results_stat)

    print(summarize_evaluation(results_stat))
    results_stat.to_csv(os.path.join(raw_results_out, f"{dataset_desc}_{perturbation_desc}-{n_perturbed_features}_{model_desc}_{computation_method}.csv"), index=False)
    np.savez(os.path.join(raw_results_out, f"{dataset_desc}_{perturbation_desc}-{n_perturbed_features}_{model_desc}_{computation_method}.npz"), raw_results)
