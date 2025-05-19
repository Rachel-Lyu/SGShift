import numpy as np
import pandas as pd
import cvxpy as cp
import shap
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.utils import check_random_state
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LassoCV
from scipy.special import expit
from sklearn.utils import resample
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from knockpy import KnockoffFilter
from knockpy import utilities
from knockpy import knockoffs
from knockpy import knockoff_stats

# ------- Classification Models -------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures

# ------- Regression Models -------
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor

import argparse

# ---------------------------------------------------------------------
# 1. Generate Source Data
#    We also return which features are truly "informative" for the
#    classification/regression tasks, to help with feature selection.
# ---------------------------------------------------------------------
def generate_source_data(task, n_samples, n_features, n_informative, random_state):
    if task == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=10,
            weights=None,
            flip_y=0.01,
            class_sep=1.0,
            hypercube=True, 
            shuffle=True,
            random_state=random_state
        )
    
    elif task == 'regression':
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            shuffle=True,
            random_state=random_state
        )

    else:
        raise ValueError("Unsupported task type. Choose 'classification' or 'regression'.")

    return X, y

# ---------------------------------------------------------------------
# 2. Fit Model
#    We support a wide range of classification and regression models.
# ---------------------------------------------------------------------
def fit_model(task, model_type, random_state, X_train = None, y_train = None):
    if task == 'classification':
        if model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(max_depth=4, random_state=random_state)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', C=1.0, random_state=random_state, probability=True)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=random_state, max_iter=200)
        elif model_type == 'mlp_classifier':
            model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500, random_state=random_state)
        elif model_type == 'adaboost_classifier':
            model = AdaBoostClassifier(n_estimators=100, random_state=random_state)
        elif model_type == 'gb_classifier':
            model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        elif model_type == 'hist_gb_classifier':
            model = HistGradientBoostingClassifier(max_iter=100, random_state=random_state)
        else:
            raise ValueError(f"Unsupported classification model type: {model_type}")

    elif task == 'regression':
        if model_type == 'knn_reg':
            model = KNeighborsRegressor(n_neighbors=5)
        elif model_type == 'decision_tree_reg':
            model = DecisionTreeRegressor(max_depth=4, random_state=random_state)
        elif model_type == 'random_forest_reg':
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif model_type == 'svm_reg':
            model = SVR(kernel='rbf', C=1.0)
        elif model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(random_state=random_state)
        elif model_type == 'lasso':
            model = Lasso(random_state=random_state)
        elif model_type == 'elasticnet':
            model = ElasticNet(random_state=random_state)
        elif model_type == 'mlp_regressor':
            model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=random_state)
        elif model_type == 'adaboost_regressor':
            model = AdaBoostRegressor(n_estimators=100, random_state=random_state)
        elif model_type == 'gb_regressor':
            model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        elif model_type == 'hist_gb_regressor':
            model = HistGradientBoostingRegressor(max_iter=100, random_state=random_state)
        else:
            raise ValueError(f"Unsupported regression model type: {model_type}")
    else:
        raise ValueError("Unsupported task type.")
    if y_train is not None:
        model.fit(X_train, y_train)
    return model

# ---------------------------------------------------------------------
# 3. Generate Target Data with Shift in P(X)
# ---------------------------------------------------------------------
def generate_target_data(X_source, shift_features=None, shift_magnitude=0., random_state=42):
    rng = check_random_state(random_state)
    X_target = X_source.copy()
    if shift_features is not None and len(shift_features) > 0:
        # For each feature in shift_features, add or subtract shift_magnitude randomly
        shift_vec = np.zeros(X_target.shape[1])
        for f_idx in shift_features:
            # We multiply shift_magnitude by either +1 or -1
            shift = shift_magnitude * rng.choice([-1, 1])
            shift_vec[f_idx] = shift
        # Broadcast-add to the entire array
        X_target = X_target + shift_vec
    return X_target

# ---------------------------------------------------------------------
# 4. Induce Conditional Shift P(y|X)
#    The user fully controls shift_subset_y and shift_magnitude_y,
#    We do NOT define them automatically. If not provided, no shift.
# ---------------------------------------------------------------------
def induce_conditional_shift(X, active_features, model, task, linear_shift_y = True, shift_subset=None, shift_magnitude=None, random_state=42):
    """
    Adjust the conditional expectation E[y | X] for either classification or regression tasks.
    
    For Classification:
        - Compute shifted log-odds: f(x) + x^T beta
        - Compute shifted probabilities: g^{-1}(f(x) + x^T beta)
        - Sample new labels based on shifted probabilities.
    
    For Regression:
        - Compute shifted expectations: g^{-1}(f(x) + x^T beta) = f(x) + x^T beta
        - Set y_shifted = shifted_expectation + noise (if desired)
    """
    rng = check_random_state(random_state)
    
    if shift_subset is None or shift_magnitude == 0:
        return model.predict(X[:, active_features])  # No shift applied
    
    if task == 'classification':
        p_S = model2p(model, X[:, active_features])
        # Compute f(x) = g(p_S) = log(p_S / (1 - p_S))
        f_x = np.log(p_S / (1 - p_S))
        
        if linear_shift_y:
            # Compute shifted log-odds: f(x) + x^T beta
            x_beta = X[:, shift_subset] @ np.array(shift_magnitude * np.random.choice([-1, 1], size=len(shift_subset)))
        else: 
            model = make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias = False), LinearRegression())
            model.fit(X[:, shift_subset], shift_magnitude * np.random.choice([-1, 1], size=X.shape[0]))
            x_beta = model.predict(X[:, shift_subset])
        shifted_log_odds = f_x + x_beta
        
        # Compute shifted probabilities: g^{-1}(shifted_log_odds) = 1 / (1 + exp(-shifted_log_odds))
        p_T = 1 / (1 + np.exp(-shifted_log_odds))
        p_T = np.clip(p_T, 1e-6, 1 - 1e-6)
        
        # Sample new labels based on shifted probabilities
        y_shifted = rng.binomial(1, p_T)
        
        return y_shifted
    
    elif task == 'regression':
        # For regression with identity link, f(x) = E_S[y | x] = model.predict(X)
        f_x = model.predict(X[:, active_features])  # Shape: (n_samples,)
        
        # Compute shifted expectation: f(x) + x^T beta
        if linear_shift_y:
            # Compute shifted log-odds: f(x) + x^T beta
            x_beta = X[:, shift_subset] @ np.array(shift_magnitude * np.random.choice([-1, 1], size=len(shift_subset)))
        else: 
            model = make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias = False), LinearRegression())
            model.fit(X[:, shift_subset], shift_magnitude * np.random.choice([-1, 1], size=X.shape[0]))
            x_beta = model.predict(X[:, shift_subset])
        shifted_expectation = f_x + x_beta
        
        # Optionally, add noise. Here, we'll assume no additional noise.
        y_shifted = shifted_expectation
        
        return y_shifted
    
    else:
        raise ValueError("Unsupported task type. Choose 'classification' or 'regression'.")

# ---------------------------------------------------------------------
# 5. Orchestrate the Entire Simulation
# ---------------------------------------------------------------------
def simulate_data(task='classification',
                  model_type='knn',
                  X_S = None, 
                  X_T_val = None,
                  y_S_tmp = None, 
                  y_T_val_freq = None,
                  n_S=800,
                  n_T=200,
                  n_val=200,
                  p=30,
                  p_informative=None,
                  shift_features_X=None,
                  shift_magnitude_X=0.,
                  shift_subset_y=None,
                  shift_magnitude_y=None,
                  linear_shift_y = True,
                  random_state=42):
    """
    Simulate data with options for different models and tasks.
    
    Parameters
    ----------
    task : str
        'classification' or 'regression'
    model_type : str
        e.g., 'knn', 'decision_tree', 'random_forest', 'svm', 'logistic_regression', 'mlp_classifier',
        'adaboost_classifier', 'gb_classifier', 'hist_gb_classifier', ...
        (for regression: 'knn_reg', 'decision_tree_reg', 'random_forest_reg', 'svm_reg',
         'linear_regression', 'ridge', 'lasso', 'elasticnet', 'mlp_regressor',
         'adaboost_regressor', 'gb_regressor', 'hist_gb_regressor', etc.)
    n_S : int
        Number of source samples.
    n_T : int
        Number of target test samples.
    n_val : int
        Number of target validation samples.
    p : int
        Number of features
    shift_features_X : list of int, optional
        Feature indices to shift in P(X). If None, no shift in P(X).
    shift_magnitude_X : float
        Magnitude of shift in P(X)
    shift_subset_y : list of int, optional
        Feature indices used to shift P(y|X). If None, no shift in P(y|X).
    shift_magnitude_y : float, optional
        Magnitude of shift in P(y|X).
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    result : dict
        {
            'source': {
                'X': ...,
                'y': ...,
            },
            'target': {
                'X': ...,
                'y': ...
            }
        }
    """
    rng = check_random_state(random_state)
    if p_informative is None:
        p_informative = p
    # 1) Generate Source Data
    # X_S, y_S_tmp = generate_source_data(task, n_S, p, p_informative, random_state)
    if X_S is None or X_T_val is None or y_S_tmp is None:
        X_total, y_total = generate_source_data(task, n_S + n_T + n_val, p, p_informative, random_state)
        X_S, y_S_tmp = X_total[:n_S], y_total[:n_S]
        X_T_val, _ = X_total[n_S:], y_total[n_S:]
        
    active_features = list(range(p_informative))
    
    # 2) (Optional) Standardize the Source Data
    scaler = StandardScaler()
    X_S = scaler.fit_transform(X_S)
    
    # 3) Fit the Chosen Model on Source Data
    model = fit_model(task, model_type, random_state, X_S[:, active_features], y_S_tmp)
    y_S = model.predict(X_S[:, active_features])

    # X_T_val, _ = generate_source_data(task, n_T + n_val, p, p_informative, random_state+1)  # re-seed for variety
    X_T_val = scaler.transform(X_T_val)
    
    # 5) Shift P(X) if requested
    if shift_features_X is not None and len(shift_features_X) > 0:
        X_T_val = generate_target_data(X_T_val, shift_features=shift_features_X,
                                shift_magnitude=shift_magnitude_X,
                                random_state=random_state)
    
    # 7) If we want to do a conditional shift in the target's labels, do it:
    y_T_val = induce_conditional_shift(
        X_T_val,
        active_features,
        model,
        task,
        linear_shift_y,
        shift_subset=shift_subset_y,
        shift_magnitude=shift_magnitude_y,
        random_state=random_state
    )
    # else:
    #     y_T_val = model.predict(X_T_val[:, active_features])
        
    X_T = X_T_val[:n_T, :]
    X_val = X_T_val[n_T:, :]
    y_T = y_T_val[:n_T]
    y_val = y_T_val[n_T:]
    
    return {
        'source': {
            'X': X_S,
            'y': y_S
        },
        'target': {
            'X': X_T,
            'y': y_T
        },
        'validation': {
            'X': X_val,
            'y': y_val
        }
    }

def get_offset_f_x(X, fitted_model=None, task='classification'):
    """
    Compute an offset f_x based on the fitted_model for either classification
    (log-odds of predicted probabilities) or regression (predicted mean).
    """
    if fitted_model is None:
        # No model offset; default to zero
        return np.zeros(X.shape[0])

    if task == 'classification':
        # model2p is assumed to return probabilities
        p_pred = model2p(fitted_model, X)
        # clip to avoid log(0)
        p_pred = np.clip(p_pred, 1e-8, 1 - 1e-8)
        f_x = np.log(p_pred / (1 - p_pred))
        return f_x
    elif task == 'regression':
        # For a regression model, we assume fitted_model.predict() gives E[Y|X]
        return fitted_model.predict(X)
    else:
        raise ValueError("task must be 'classification' or 'regression'.")

def loss_function(linear_predictor, y, task='classification'):
    """
    Returns the total (un-averaged) loss for either classification (logistic deviance)
    or regression (sum of squares).
    
    Classification: sum(log(1 + exp(lp))) - y^T lp
    Regression: sum((lp - y)^2)
    """
    if task == 'classification':
        # logistic(dev) = \sum log(1 + exp(lp)) - y * lp
        return cp.sum(cp.logistic(linear_predictor)) - y @ linear_predictor
    elif task == 'regression':
        # squared error
        return cp.sum_squares(linear_predictor - y)
    else:
        raise ValueError("task must be 'classification' or 'regression'.")

def estimate_delta(X_T, 
                   y_T, 
                   lambda_reg, 
                   fitted_model=None, 
                   f_x=None, 
                   task='classification'):
    """
    Estimates the sparse parameter delta using the transfer learning proposal.

    Parameters
    ----------
    X_T : array-like of shape (n_T, p)
    y_T : array-like of shape (n_T,)
    lambda_reg : float
    fitted_model : optional, trained model (used for offset f_x)
    f_x : optional, precomputed offset
    task : {'classification', 'regression'}

    Returns
    -------
    delta_est : array of shape (p,)
    """
    n_T, p = X_T.shape
    delta = cp.Variable(p)

    # Define the linear predictor: f(x) + X * delta
    if f_x is None:
        f_x = get_offset_f_x(X_T, fitted_model, task=task)
    linear_predictor = f_x + X_T @ delta

    # Define the loss
    raw_loss = loss_function(linear_predictor, y_T, task=task)
    # Average the loss over samples for consistency
    loss_avg = raw_loss / n_T

    # L1 regularization
    l1_penalty = lambda_reg * cp.norm1(delta)

    # Total objective
    total_loss = loss_avg + l1_penalty

    # Solve
    problem = cp.Problem(cp.Minimize(total_loss))
    problem.solve(solver=cp.SCS, verbose=False)

    if delta.value is not None:
        return delta.value
    else:
        raise ValueError("Optimization did not converge.")

def estimate_beta_delta(X_S, 
                        y_S, 
                        X_T, 
                        y_T, 
                        lambda_beta, 
                        lambda_delta, 
                        fitted_model=None, 
                        f_x=None, 
                        task='classification'):
    """
    Estimates sparse parameters beta and delta for transfer learning.

    Parameters
    ----------
    X_S : array (n_S, p)
    y_S : array (n_S,)
    X_T : array (n_T, p)
    y_T : array (n_T,)
    lambda_beta : float
    lambda_delta : float
    fitted_model : optional, used to get offset
    f_x : optional, precomputed offset
    task : {'classification', 'regression'}

    Returns
    -------
    beta_est, delta_est : arrays of shape (p,)
    """
    n_S, p = X_S.shape
    n_T = X_T.shape[0]

    # Construct X' = [X_S, 0; X_T, X_T]
    zero_S = np.zeros((n_S, p))
    X_prime = np.vstack([
        np.hstack([X_S, zero_S]),
        np.hstack([X_T, X_T])
    ])
    # Stack y' = [y_S; y_T]
    y_prime = np.concatenate([y_S, y_T])

    # Define variable
    beta_prime = cp.Variable(2 * p)

    # Offset f_x
    if f_x is None:
        # f_x for the stacked data
        X_stack = np.vstack([X_S, X_T])
        f_x = get_offset_f_x(X_stack, fitted_model, task=task)
    # linear predictor
    eta_prime = f_x + X_prime @ beta_prime

    # Loss
    raw_loss = loss_function(eta_prime, y_prime, task=task)
    loss_avg = raw_loss / len(y_prime)

    # Separate L1 penalties
    beta_l1 = cp.norm1(beta_prime[:p])
    delta_l1 = cp.norm1(beta_prime[p:])

    total_loss = loss_avg + lambda_beta * beta_l1 + lambda_delta * delta_l1

    # Solve
    problem = cp.Problem(cp.Minimize(total_loss))
    problem.solve(solver=cp.SCS, verbose=False)

    if beta_prime.value is not None:
        beta_est = beta_prime.value[:p]
        delta_est = beta_prime.value[p:]
        return beta_est, delta_est
    else:
        raise ValueError("Optimization did not converge.")

def cross_validate_lambda(X_T, 
                          y_T, 
                          fitted_model, 
                          lambdas, 
                          misspecified, 
                          X_S=None, 
                          y_S=None, 
                          k=5, 
                          task='classification'):
    """
    Cross-validation to select the optimal lambda.

    For classification: uses a logistic deviance + penalty approach and BIC.
    For regression: uses squared error + BIC.

    Parameters
    ----------
    ...
    task : {'classification', 'regression'}
    ...
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mean_losses = []

    for lambda_reg in lambdas:
        fold_losses = []
        for train_index, val_index in kf.split(X_T):
            X_train, X_val = X_T[train_index], X_T[val_index]
            y_train, y_val = y_T[train_index], y_T[val_index]
            
            # Compute offset on validation set
            f_x_val = get_offset_f_x(X_val, fitted_model, task=task)

            if misspecified:
                beta_estimated, delta_estimated = estimate_beta_delta(
                    X_S, y_S, X_train, y_train,
                    0.9 * lambda_reg, lambda_reg, 
                    fitted_model=fitted_model, 
                    task=task
                )
                linear_predictor_val = f_x_val + X_val @ (beta_estimated + delta_estimated)

                # number of nonzero coefficients
                k_nonzero = np.sum(np.abs(np.concatenate([beta_estimated, delta_estimated])) > 1e-4)
            else:
                # The simpler approach, only delta
                delta_estimated = estimate_delta(
                    X_train, y_train, lambda_reg, 
                    fitted_model=fitted_model, 
                    task=task
                )
                linear_predictor_val = f_x_val + X_val @ delta_estimated
                k_nonzero = np.sum(np.abs(delta_estimated) > 1e-4)

            # Compute deviance or RSS
            if task == 'classification':
                # dev = sum(log(1 + exp(lp))) - y_val @ lp
                dev_val = np.sum(np.log(1 + np.exp(linear_predictor_val))) - y_val @ linear_predictor_val
                # BIC
                # 2 * dev + k * log(n)
                if k_nonzero == 0:
                    BIC_loss = np.inf
                else:
                    BIC_loss = 2.0 * dev_val + k_nonzero * np.log(len(y_val))
            else:
                # regression
                # RSS = sum((lp - y_val)^2)
                rss_val = np.sum((linear_predictor_val - y_val)**2)
                # BIC
                if k_nonzero == 0:
                    BIC_loss = np.inf
                else:
                    BIC_loss = len(y_val) * np.log(rss_val / len(y_val) + 1e-12) + k_nonzero * np.log(len(y_val))

            fold_losses.append(BIC_loss)
        mean_losses.append(np.mean(fold_losses))

    # find best lambda
    optimal_lambda_index = np.argmin(mean_losses)
    optimal_lambda = lambdas[optimal_lambda_index]

    # re-estimate final
    if misspecified:
        optimal_beta, optimal_delta = estimate_beta_delta(
            X_S, y_S, X_T, y_T,
            0.9 * optimal_lambda, optimal_lambda, 
            fitted_model=fitted_model, 
            task=task
        )
        return optimal_beta, optimal_delta, optimal_lambda
    else:
        optimal_delta = estimate_delta(
            X_T, y_T, optimal_lambda, 
            fitted_model=fitted_model, 
            task=task
        )
        return optimal_delta, optimal_lambda

def select_beta_knockoff(X_T, 
                         y_T, 
                         f_x, 
                         lambda_reg, 
                         task='classification'):
    """
    Knockoff-based selection for the (single-task) scenario.

    Parameters
    ----------
    ...
    task : {'classification', 'regression'}
    ...
    """
    # Create knockoffs
    kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
    Sigma, _ = utilities.estimate_covariance(X_T, 1e-2, "ledoitwolf")
    kfilter.X = X_T

    kfilter.ksampler = knockoffs.GaussianSampler(X=X_T, groups=None, mu=None)
    kfilter.Xk = kfilter.ksampler.sample_knockoffs()

    n_T, p = X_T.shape
    beta = cp.Variable(2 * p)

    # Combine original and knockoffs side by side, then random permutation
    features = np.concatenate([kfilter.X, kfilter.Xk], axis=1)
    inds, rev_inds = utilities.random_permutation_inds(2 * p)
    features = features[:, inds]

    # Define linear predictor: f_x + features @ beta
    linear_predictor = f_x + features @ beta

    # The loss depends on classification or regression
    raw_loss = loss_function(linear_predictor, y_T, task=task)
    loss_avg = raw_loss / len(y_T)

    # L1 penalty
    l1_penalty = lambda_reg * cp.norm1(beta)
    total_loss = loss_avg + l1_penalty

    problem = cp.Problem(cp.Minimize(total_loss))
    problem.solve(solver=cp.SCS, verbose=False)

    Z = beta.value[rev_inds]

    # Combine Z stats for final W
    W = knockoff_stats.combine_Z_stats(Z)
    return W

def select_beta_knockoff_misspecified(X_S, 
                                      y_S, 
                                      X_T, 
                                      y_T, 
                                      f_x, 
                                      lambda_beta, 
                                      lambda_delta, 
                                      task='classification'):
    """
    Knockoff-based selection for the misspecified scenario (transfer).
    """
    kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
    Sigma, _ = utilities.estimate_covariance(X_T, 1e-2, "ledoitwolf")
    kfilter.X = X_T
    
    n_S, p = X_S.shape
    n_T = X_T.shape[0]
    
    kfilter.ksampler = knockoffs.GaussianSampler(X=X_T, groups=None, mu=None)
    kfilter.Xk = kfilter.ksampler.sample_knockoffs()
    
    # beta_prime is size 3*p in the original code
    beta_prime = cp.Variable(3 * p)

    # Combine X, Xk for T
    features = np.concatenate([kfilter.X, kfilter.Xk], axis=1)
    inds, rev_inds = utilities.random_permutation_inds(2 * p)
    features = features[:, inds]

    # Construct X' = [X_S, 0; X_T, features]
    zero_S = np.zeros((n_S, 2 * p))
    X_prime = np.vstack([
        np.hstack([X_S, zero_S]),
        np.hstack([X_T, features])
    ])
    y_prime = np.concatenate([y_S, y_T])

    # linear predictor
    linear_predictor = f_x + X_prime @ beta_prime

    raw_loss = loss_function(linear_predictor, y_prime, task=task)
    loss_avg = raw_loss / len(y_prime)

    # Separate L1 penalties for beta (first p) and delta (remaining 2p)
    beta_l1 = cp.norm1(beta_prime[:p])
    delta_l1 = cp.norm1(beta_prime[p:])

    total_loss = loss_avg + lambda_beta * beta_l1 + lambda_delta * delta_l1

    problem = cp.Problem(cp.Minimize(total_loss))
    problem.solve(solver=cp.SCS, verbose=False)

    # Z stats are from the "delta portion"
    Z = beta_prime.value[p:][rev_inds]
    W = knockoff_stats.combine_Z_stats(Z)
    
    return W

def derandom_knock_misspecified(solver, 
                                X_S, 
                                y_S, 
                                X_T, 
                                y_T, 
                                B, 
                                lambdas, 
                                v=3, 
                                task='classification'):
    """
    De-randomized knockoffs for the misspecified scenario.

    Parameters
    ----------
    ...
    task : {'classification', 'regression'}
    ...
    """

    n_S, p = X_S.shape
    curr_W = {lambda_reg: [] for lambda_reg in lambdas}
    pi = {lambda_reg: np.zeros(p) for lambda_reg in lambdas}

    for b in range(B):
        # Resample half of the source data
        X_S_b, y_S_b = resample(X_S, y_S, n_samples=n_S//2, random_state=42 + b)
        X_b = np.vstack([X_S_b, X_T])
        # Fit solver on X_S_b, y_S_b
        solver.fit(X_S_b, y_S_b)
        
        # Compute offset for the entire stacked data (S_b + T)
        f_x = get_offset_f_x(X_b, solver, task=task)
        
        for lambda_reg in lambdas:
            lambda_beta = 0.9 * lambda_reg
            lambda_delta = lambda_reg
            # select with knockoff
            W = select_beta_knockoff_misspecified(
                X_S_b, y_S_b, X_T, y_T, 
                f_x, lambda_beta, lambda_delta,
                task=task
            )
            curr_W[lambda_reg].append(W)

    # Now aggregate
    for lambda_reg in lambdas:
        for W in curr_W[lambda_reg]:
            if np.sum(W < 0) < v:
                S = np.where(W > 0)[0] 
                pi[lambda_reg][S] += 1
            else:
                order_w = np.argsort(np.abs(W))[::-1]
                sorted_w = W[order_w]
                negid = np.where(sorted_w < 0)[0]
                # We want the (v-1)-th negative
                if len(negid) < v:
                    # fallback if fewer than v negatives
                    S = np.where(W > 0)[0]
                    pi[lambda_reg][S] += 1
                    continue
                TT = negid[v - 1]
                S = np.where(sorted_w[:TT] > 0)[0]
                S = order_w[S]
                pi[lambda_reg][S] += 1
        pi[lambda_reg] /= len(curr_W[lambda_reg])

    return pi

def derandom_knock(solver, 
                   X_S, 
                   y_S, 
                   X_T, 
                   y_T, 
                   B, 
                   lambdas, 
                   v=3, 
                   task='classification'):
    """
    De-randomized knockoffs for single-task.

    Parameters
    ----------
    ...
    task : {'classification', 'regression'}
    ...
    """

    n_S, p = X_S.shape
    curr_W = {lambda_reg: [] for lambda_reg in lambdas}
    pi = {lambda_reg: np.zeros(p) for lambda_reg in lambdas}

    for b in range(B):
        # Resample half
        X_S_b, y_S_b = resample(X_S, y_S, n_samples=n_S//2, random_state=42 + b)
        
        # Fit
        solver.fit(X_S_b, y_S_b)
        
        # Offset on X_T
        f_x = get_offset_f_x(X_T, solver, task=task)
        
        for lambda_reg in lambdas:
            W = select_beta_knockoff(X_T, y_T, f_x, lambda_reg=lambda_reg, task=task)
            curr_W[lambda_reg].append(W)

    # Aggregate
    for lambda_reg in lambdas:
        for W in curr_W[lambda_reg]:
            if np.sum(W < 0) < v:
                S = np.where(W > 0)[0] 
                pi[lambda_reg][S] += 1
            else:
                order_w = np.argsort(np.abs(W))[::-1]
                sorted_w = W[order_w]
                negid = np.where(sorted_w < 0)[0]
                if len(negid) < v:
                    # fallback
                    S = np.where(W > 0)[0]
                    pi[lambda_reg][S] += 1
                    continue
                TT = negid[v - 1]
                S = np.where(sorted_w[:TT] > 0)[0]
                S = order_w[S]
                pi[lambda_reg][S] += 1
        pi[lambda_reg] /= len(curr_W[lambda_reg])
    return pi

def model2p(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    else:
        p = expit(solver.decision_function(X))
    p[p > 1-1e-6] = 1-1e-6
    p[p < 1e-6] = 1e-6
    return p

def evaluation(y_val, y_pred, true = None, selected = None, task = 'classification'):
    if task == 'classification':
        mask = ~np.isnan(y_pred)
        ll = log_loss(y_val[mask], y_pred[mask])
        auc = roc_auc_score(y_val, y_pred)
        if true is not None:
            true = set(true)
            selected = set(selected)
            tp = len(selected & true)  # True Positives
            fp = len(selected - true)  # False Positives
            fn = len(true - selected)  # False Negatives
            return {'log_loss': ll,
                    'AUC': auc,
                    'TP': tp,
                    'FP': fp,
                    'FN': fn}
        else:
            return {'log_loss': ll,
                    'AUC': auc}
    elif task == 'regression':
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        if true is not None:
            true = set(true)
            selected = set(selected)
            tp = len(selected & true)  # True Positives
            fp = len(selected - true)  # False Positives
            fn = len(true - selected)  # False Negatives
            return {'MSE': mse,
                    'MAE': mae,
                    'TP': tp,
                    'FP': fp,
                    'FN': fn}
        else:
            return {'MSE': mse,
                    'MAE': mae}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulation")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], default='classification',
                        help='Task type: classification or regression.')
    class_models = [
        'knn', 'decision_tree', 'random_forest', 'svm',
        'logistic_regression', 'mlp_classifier',
        'adaboost_classifier', 'gb_classifier', 'hist_gb_classifier'
    ]
    reg_models = [
        'knn_reg', 'decision_tree_reg', 'random_forest_reg', 'svm_reg',
        'linear_regression', 'ridge', 'lasso', 'elasticnet', 'mlp_regressor',
        'adaboost_regressor', 'gb_regressor', 'hist_gb_regressor'
    ]
    all_models = class_models + reg_models
    parser.add_argument('--model', type=str, choices=all_models, default='logistic_regression',
                        help='Which generative model.')
    parser.add_argument('--solver', type=str, choices=all_models, default='logistic_regression',
                        help='Which solver to use.')
    parser.add_argument('--B', type=int, default=1000,
                        help='Number of repeats.')
    parser.add_argument('--n_test', type=int, default=1000,
                        help='Number of test samples.')
    parser.add_argument('--shift_magnitude_y', type=float, default=2.,
                        help='Magnitude of shift in P(y|X).')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--directory', type=str, default='out',
                        help='Output directory.')
    return parser.parse_args()

def matchsample(X, y, random_seed=42):
    # Split the majority and minority classes
    X_minority = X[y == 1]
    X_majority = X[y == 0]

    y_minority = y[y == 1]
    y_majority = y[y == 0]

    # Downsample majority class
    X_majority_downsampled, y_majority_downsampled = resample(
        X_majority, y_majority, 
        replace=False,  # Without replacement
        n_samples=len(y_minority),  # Match the minority class count
        random_state=random_seed  # Set seed for reproducibility
    )

    # Combine minority and downsampled majority
    X_balanced = np.vstack((X_minority, X_majority_downsampled))
    y_balanced = np.hstack((y_minority, y_majority_downsampled))

    # Shuffle dataset
    perm = np.random.RandomState(random_seed).permutation(len(y_balanced))
    return X_balanced[perm], y_balanced[perm]

def downsample(X, y, n_samples, random_seed=42):
    classes, counts = np.unique(y, return_counts=True)

    # Identify majority and minority class labels
    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]

    # Split the data
    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]
    X_majority = X[y == majority_class]
    y_majority = y[y == majority_class]
    n_minority = len(y_minority)
    if n_samples <= 2 * n_minority:
        n_majority = n_minority
    else: 
        n_majority = n_samples - n_minority
        
    X_majority_downsampled, y_majority_downsampled = resample(
        X_majority, y_majority, 
        replace=False,  # Without replacement
        n_samples=n_minority,  # Match the minority class count
        random_state=random_seed  # Set seed for reproducibility
    )

    X_balanced = np.vstack((X_minority, X_majority_downsampled))
    y_balanced = np.hstack((y_minority, y_majority_downsampled))

    perm = np.random.RandomState(random_seed).permutation(len(y_balanced))
    X_new = X_balanced[perm]
    y_new = y_balanced[perm]
    if n_samples > 2 * n_minority:
        return X_new, y_new
    else:
        X_downsampled, y_downsampled = resample(
            X_new, y_new, 
            replace=False,  # Without replacement
            n_samples=n_samples,  # Match the minority class count
            random_state=random_seed  # Set seed for reproducibility
        )
        return X_downsampled, y_downsampled

def fit_nonparametric_classifier_with_offset(X_T, y_T, f_x, selected_features=None, lambda_reg = 1.0, **kwargs):
    """
    Fits a nonparametric classifier (for discrete outcomes) using the source model's 
    predictions (f_x) as an offset (with coefficient fixed to 1) and the selected subset of features from X_T.
    
    Parameters
    ----------
    X_T : array-like of shape (n_T, p)
          The target domain feature matrix.
    y_T : array-like of shape (n_T,)
          The target domain labels.
    f_x : array-like of shape (n_T,)
          The predictions from the source model to be used as an offset.
    selected_features : list or array of int, optional
          Indices of the features in X_T selected as shifted. If None, all features are used.
    **kwargs : additional keyword arguments
          Passed to the GLM fitting routine.
    
    Returns
    -------
    result : statsmodels GLMResults instance
    """
    # Subset X_T if selected_features is provided; otherwise, use all features.
    if selected_features is not None:
        X_sel = X_T[:, selected_features]
    else:
        X_sel = X_T

    # Ensure f_x is a numpy array and flatten it to 1D (it will be used as an offset)
    f_x = np.array(f_x).flatten()
    
    degree = kwargs.pop('degree', 2)
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_design = poly.fit_transform(X_sel)
    
    # Optionally add a constant to the design matrix if desired.
    if kwargs.pop('add_const', True):
        X_design = sm.add_constant(X_design, has_constant='add')
    
    # Choose the appropriate family based on the outcome.
    if np.unique(y_T).size == 2:
        family = sm.families.Binomial()
    else:
        family = sm.families.Gaussian()
    
    # Fit the GLM with the given offset.
    model = sm.GLM(y_T, X_design, family=family, offset=f_x)
    model = model.fit_regularized(alpha=lambda_reg, L1_wt=1.0)
    return model