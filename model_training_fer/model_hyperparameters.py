"""Hyperparameters for the models."""
hyperparameters_dict = {
    "logistic_regression": {
        # "C": [0.1, 1, 10, 100, 1000],
        "penalty": ["l1", "l2", "elasticnet", None],
        "solver": ["saga", "lbfgs", "newton-cg", "sag"],
        # "tol": [1e-4, 1e-3, 1e-2, 1],
        "n_jobs": [-1]
    },
    "naive_bayes": {
        "alpha": [0.1, 1, 10, 100, 1000],
        "fit_prior": [True, False]
    },
    "random_forest": {
        # "n_estimators": [10, 100, 1000],
        "criterion": ["gini", "entropy"],
        # "min_samples_split": [2, 10, 100, 1000],
        # "min_samples_leaf": [1, 10, 100, 1000],
        "max_features": ["auto", "sqrt", "log2"],
        "n_jobs": [-1]
    },
    "xgboost": {
        "n_estimators": [10, 100, 1000],
        # "max_depth": [3, 5, 7, 9],
        # "learning_rate": [0.01, 0.1, 0.3, 0.5],
        "subsample": [0.5, 0.7, 1],
        "colsample_bytree": [0.5, 0.7, 1],
        "n_jobs": [-1]
    }
}