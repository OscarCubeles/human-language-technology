from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os

import matplotlib.pyplot as plt


def identify_high_correlations(features, threshold=0.7):
    corr_values = features.corr()
    cols = features.columns
    correlated_pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_value = corr_values.iloc[i, j]

            if abs(corr_value) > threshold:
                correlated_pairs.append((
                    cols[i],              # feature 1 name
                    cols[j],              # feature 2 name
                    corr_value            # actual correlation
                ))

    return correlated_pairs
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy

import numpy as np
from sklearn.base import clone
# Generate all combinations manually
from itertools import product

def run_grid_search(model, param_grid, X_train, y_train, X_val, y_val, verbose=0):
    """
    Custom grid search WITHOUT cross-validation.
    Trains one model per parameter combination using X_train only.
    Selects the best model based on validation R on X_val.
    """
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_param_combinations = list(product(*param_values))

    best_score = -np.inf
    best_params = None
    best_model = None

    for i, combo in enumerate(all_param_combinations):

        params = dict(zip(param_keys, combo))

        if verbose:
            print(f"Testing params ({i+1}/{len(all_param_combinations)}): {params}")

        # Create a fresh model with these hyperparameters
        model_instance = clone(model).set_params(**params)

        # Train on training set
        model_instance.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = model_instance.predict(X_val)
        val_r = np.corrcoef(y_val, y_pred)[0, 1]

        if verbose:
            print(f"    Validation R: {val_r:.4f}")

        # Keep the best
        if val_r > best_score:
            best_score = val_r
            best_params = params
            best_model = model_instance

    return best_model, best_score, best_params


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import clone
import numpy as np

def pearson_r(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

pearson_scorer = make_scorer(pearson_r, greater_is_better=True)


def run_grid_search_cv(model, param_grid, X, y, verbose=0, cv=5):
    """
    Grid search using sklearn's GridSearchCV with Pearson correlation.
    Returns:
        - best_model: estimator refit on full training data
        - best_score: mean CV Pearson correlation
        - best_params: best hyperparameters
    """

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=pearson_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
        refit=True   # ensures best_model is fitted on full X,y
    )

    gs.fit(X, y)

    best_model = gs.best_estimator_
    best_score = gs.best_score_          # mean CV Pearson R
    best_params = gs.best_params_

    return best_model, best_score, best_params

import pandas as pd

def save_to_csv(
    out_file,
    all_features,
    active_features,
    feature_removed,
    best_iter_score,
    best_global_score,
    first_write=False
):
    """
    Append a row to the feature elimination CSV log.
    
    first_write = True → overwrites existing file and writes header
    """
    
    # binary mask for which features are active
    feature_mask = {
        feat: (1 if feat in active_features else 0)
        for feat in all_features
    }

    # add extra info
    feature_mask["feature_removed"] = feature_removed
    feature_mask["best_iter_score"] = best_iter_score
    feature_mask["best_global_score"] = best_global_score

    # convert to 1-row dataframe
    df_row = pd.DataFrame([feature_mask])

    # write header if first_write=True
    if first_write:
        df_row.to_csv(out_file, index=False)
    else:
        df_row.to_csv(out_file, mode="a", header=False, index=False)

def backward_elimination(
    X_train, y_train,
    base_model,
    param_grid,
    X_val=None, y_val=None,
    out_file=None,
    verbose=1
): 
    """
    Custom backward feature elimination with clear progress printing.
    At each iteration tests all the subsets of features (removing one at a time) and 
    keeps the best subset of features.
    Then does the same starting from the current feature subset.

    By default, cross validation will be used to determine the performance of each model,
      splitting X_train and y_train and exploring the param grid
    But if X_val and y_val are provided, simple hold out validation on the provided val set will be done
    """
    assert (X_val is None and y_val is None) or not (X_val is None or y_val is None)
    n_features = X_train.shape[1]
    active_features = list(X_train.columns)
    print(active_features)

    best_global_score = -np.inf
    best_global_features = None
    best_global_model = None
    best_global_params = None
    if out_file is None: 
        out_file = 'data/feature_selection/' + base_model.__class__.__name__ + '.csv'
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    iteration = 0
    removed_features_history = []
    first_write = True
    while len(active_features) > 1:

        iteration += 1
        if verbose:
            print("=" * 80)
            print(f"Iteration {iteration} / {n_features-1}")
            print("=" * 80)

        best_iter_score = -np.inf
        best_iter_subset = None
        best_iter_model = None
        best_iter_params = None
        last_feature_removed = None

        # Try removing each feature

        for j, f in enumerate(active_features):
            subset = [x for x in active_features if x != f]

            if verbose:
                print(f"\tTesting removal of feature {f}")
                print(f"\tfeature {j+1} / {len(active_features)} ")

            # Subset data
            X_train_sub = X_train[subset]
            if X_val is not None:
                X_val_sub = X_val[subset]
                # grid exploration with hold out validation
                model, r_val, params = run_grid_search(
                    base_model,
                    param_grid,
                    X_train_sub,
                    y_train,
                    X_val_sub,
                    y_val,
                    verbose=0
                )
            else: 
                # grid exploration with cross validation
                model, r_val, params = run_grid_search_cv(
                    base_model,
                    param_grid,
                    X_train_sub,
                    y_train,
                    verbose=0
                )
                

            if verbose:
                print(f"    R Score = {r_val:.4f}, ")

            # Track best in this iteration
            if r_val> best_iter_score:
                best_iter_score = r_val
                best_iter_subset = subset
                last_feature_removed = f
                best_iter_model = model
                best_iter_params = params
        
        # Print iteration summary
        if verbose:
            print("\n✔ Best subset in this iteration:", best_iter_subset)
            print("removed ", last_feature_removed )
            print(f"✔ Best validation Score in iteration: {best_iter_score:.4f}")
            print("-" * 80)

        # Update global best
        if best_iter_score > best_global_score:
            best_global_score = best_iter_score
            best_global_features = deepcopy(best_iter_subset)
            best_global_model = deepcopy(best_iter_model)
            best_global_params = deepcopy(best_iter_params)

        # Reduce feature set
        active_features = best_iter_subset
        removed_features_history.append((
            last_feature_removed, active_features, best_iter_score, best_global_score))


        # Write one CSV row for this iteration
        save_to_csv(
            out_file=out_file,
            all_features=list(X_train.columns),
            active_features=active_features,
            feature_removed=last_feature_removed,
            best_iter_score=best_iter_score,
            best_global_score=best_global_score,
            first_write=first_write
        )

        first_write = False  # after first iteration, always append

    return {
        "best_features": best_global_features,
        "best_model": best_global_model,
        "best_params": best_global_params,
        "best_score": best_global_score,
        "history":removed_features_history
    }


def plot_elimination_history(history, title):
    iters = range(len(history))
    scores = history.best_iter_score        # best_iter_score
    removed = history.feature_removed

    plt.figure(figsize=(12, 6))
    plt.plot(iters, scores, marker="o")
    
    for i, feat in enumerate(removed):
        plt.text(iters[i], scores[i], feat, fontsize=8, rotation=45)

    plt.xlabel("Iteration")
    plt.ylabel("Best Validation Pearson R")
    plt.title(title)
    plt.grid(axis="both", linestyle="--", alpha=0.4)
    plt.show()



def identify_bad_predictions( df_sentences, y_pred, method="absolute", threshold=None, top_k=None):
    """
    Identify samples where the prediction is far from the true value.
    """

    y_true = df_sentences.score.to_numpy()
    y_pred = np.array(y_pred)

    if method == "absolute":
        errors = np.abs(y_true - y_pred)
    elif method == "squared":
        errors = (y_true - y_pred)**2
    else:
        raise ValueError("method must be 'absolute' or 'squared'.")

    df = pd.DataFrame({
        "true": y_true,
        "pred": y_pred,
        "error": errors,
        'sent1':df_sentences.sentence1,
        "sent2":df_sentences.sentence2
    })

    # Option 1: filter by an explicit error threshold
    if threshold is not None:
        df = df[df["error"] >= threshold]

    # Option 2: return the k biggest errors
    if top_k is not None:
        df = df.sort_values("error", ascending=False).head(top_k)

    return df.sort_values("error", ascending=False)
