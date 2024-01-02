import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.pipeline import Pipeline


class Tuner:
    def __init__(self, preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series):
        self.preprocessor = preprocessor
        self.X_train = X_train
        self.y_train = y_train

    def tune_random_forest_regressor(self, rfr_pipe: Pipeline, d: int = None, verbose: bool = False) -> Pipeline:

        if d is None:
            d = int(self.X_train.shape[1])

        parameters_rfr = {
            "randomforestregressor__max_depth": np.arange(int(np.floor(np.sqrt(d)/2)), int(np.floor(np.sqrt(d)*2))),
            "randomforestregressor__n_estimators": np.arange(20, 100)
        }
        rs_rfr = RandomizedSearchCV(rfr_pipe, parameters_rfr, scoring="neg_mean_absolute_error")
        rs_rfr.fit(self.X_train, np.ravel(self.y_train))

        max_depth_best = rs_rfr.best_params_["randomforestregressor__max_depth"]
        n_estimators_best = rs_rfr.best_params_["randomforestregressor__n_estimators"]

        if verbose:
            print(f"RFR Best maximum depth: {max_depth_best}")
            print(f"RFR Best number of estimators: {n_estimators_best}")

        tuned_rfr_pipe = make_pipeline(
            self.preprocessor,
            StandardScaler(),
            RandomForestRegressor(max_depth=max_depth_best, n_estimators=n_estimators_best)
        )

        return tuned_rfr_pipe

    def tune_lasso(
            self,
            lasso_pipe: Pipeline,
            verbose: bool = False
    ) -> Pipeline:

        parameters_lasso = {"lasso__alpha": np.linspace(0.1, 5, 10)}  # Some values for the regularization strength
        gs_lasso = GridSearchCV(lasso_pipe, parameters_lasso, scoring="neg_mean_absolute_error")

        # Since LASSO is fast, we can perform a grid search for the optimal alpha
        gs_lasso.fit(self.X_train, self.y_train)
        alpha_best_lasso = gs_lasso.best_params_["lasso__alpha"]
        if verbose:
            print("\nGrid Search Results:\n")
            print(f"Optimal alpha: {alpha_best_lasso}")
            print(pd.DataFrame(gs_lasso.cv_results_).head(5))

        # Update the LASSO pipeline with the optimal alpha
        lasso_pipe = make_pipeline(self.preprocessor, StandardScaler(), Lasso(alpha=alpha_best_lasso))

        return lasso_pipe

    def tune_ensemble(self, pipe_lasso: Pipeline, pipe_rfr: Pipeline, pipe_svr: Pipeline, verbose: bool = False):

        # Ensemble Model (Stacking)
        sr_ridge = StackingRegressor(
            estimators=[
                ("lasso", pipe_lasso),
                ("rfr", pipe_rfr),
                ("svr", pipe_svr)
            ],
            final_estimator=Ridge(alpha=1)
        )

        if verbose:
            print("Mean of cross validation results - stacking model:")
            print(np.mean(pd.DataFrame(
                cross_validate(sr_ridge, self.X_train, np.ravel(self.y_train), scoring="neg_mean_absolute_error"))))

        return sr_ridge
