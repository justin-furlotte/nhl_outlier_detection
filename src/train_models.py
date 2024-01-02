from utils.constants import dtypes
from utils.preprocessing import Preprocessor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from utils.hyperparameter_tuning import Tuner
import pandas as pd
import numpy as np
import pickle
pd.set_option('expand_frame_repr', False)  # Display full dataframes while printing them
from utils.data_loading import load_data
from src.predict import predict_from_pickled_models


def train_and_save_models(X_train: pd.DataFrame, y_train: pd.Series, target: str, situation: str) -> dict:

    cleaner = Preprocessor()
    preprocessor = cleaner.create_preprocessor(X_train=X_train)

    tuner = Tuner(preprocessor=preprocessor, X_train=X_train, y_train=y_train)

    # ---------------- Create a pipeline for some models ----------------

    # LASSO Regression
    lasso_pipe = make_pipeline(preprocessor, StandardScaler(), Lasso(max_iter=1000))
    tuned_lasso_pipe = tuner.tune_lasso(lasso_pipe=lasso_pipe, verbose=True)

    # Random forest regression
    rfr_pipe = make_pipeline(preprocessor, RandomForestRegressor())
    tuned_rfr_pipe = tuner.tune_random_forest_regressor(rfr_pipe=rfr_pipe, verbose=True)

    # SVM Regression
    # Unfortunately it is too slow to do hyperparameter optimization on SVR, but it is still giving
    # decent results with a regularization strength of 1 and Gaussian RBF kernel
    svr_pipe = make_pipeline(preprocessor, StandardScaler(), SVR(kernel="rbf", C=1.0))

    # Create and tune a stacking ensemble model with Ridge as the final regressor.
    ensemble = tuner.tune_ensemble(pipe_lasso=tuned_lasso_pipe, pipe_rfr=tuned_rfr_pipe, pipe_svr=svr_pipe, verbose=True)

    # save models
    print("Saving models...\n")
    with open(f"../models/LASSO_pickle_{target}_{situation}_{train_start}_{train_end}", "wb") as f:
        pickle.dump(tuned_lasso_pipe, f)
    with open(f"../models/RFR_pickle_{target}_{situation}_{train_start}_{train_end}", "wb") as f:
        pickle.dump(tuned_rfr_pipe, f)
    with open(f"../models/SVR_pickle_{target}_{situation}_{train_start}_{train_end}", "wb") as f:
        pickle.dump(svr_pipe, f)
    with open(f"../models/SR_pickle_{target}_{situation}_{train_start}_{train_end}", "wb") as f:
        pickle.dump(ensemble, f)

    models = {
        "LASSO": tuned_lasso_pipe,
        "RFR": tuned_rfr_pipe,
        "SVR": svr_pipe,
        "Ensemble": ensemble
    }

    return models


if __name__ == "__main__":

    # File path for the dataset
    base_fp = "../data"
    fp = f"{base_fp}/Players_All_Years.csv"

    # Minimum number of games a player must have played in to be eligible for modelling.
    min_games = 30

    # Define the target variable and the "situation" (i.e. 5on5, PP, PK, all)
    targets = ["I_F_goals", "I_F_points", "I_F_primaryAssists", "I_F_secondaryAssists"]
    situations = ["all", "5on5", "4on5", "5on4"]

    # Beginning and end of the training data (inclusive). Note that "2010" represents the 2010-2011 season, for example.
    train_start = 2010
    train_ends = [2011 + i for i in range(12)]

    for target in targets:
        for situation in situations:
            for train_end in train_ends:

                # Load the data
                df = load_data(fp=fp)

                df = df.loc[df["situation"] == situation]  # Restrict to only the desired situation
                df = df.loc[df["games_played"] >= min_games]  # Restrict to only a minimum number of games played

                # List of seasons used for the train set, inclusive (recall e.g. 2020 means the 2020-2021 season)
                train_seasons = [train_start + i for i in range(train_end - train_start + 1)]

                # Define the train and test sets
                df_train = df.loc[df.index.get_level_values("season").isin(train_seasons)]
                df_test = df.loc[df.index.get_level_values("season") == train_end + 1]

                X_train = df_train.drop(target, axis=1)
                y_train = df_train[target]
                X_test = df_test.drop(target, axis=1)
                y_test = df_test[target]

                # Train the models, tune the hyperparameters, and pickle the tuned models
                models = train_and_save_models(X_train=X_train, y_train=y_train, situation=situation, target=target)

    # Now that all the models' hyperparameters have been appropriately tuned, we can make and save the predictions
    # for each combination of hockey season, scoring situation, and target variable..
    predict_from_pickled_models()

    # ensemble_coefs = ensemble.final_estimator_.coef_
    # print(f"Stacking estimator (Ridge) coefficients:")
    # print(f"LASSO: {ensemble_coefs[0]}")
    # print(f"RFR: {ensemble_coefs[1]}")
    # print(f"SVR: {ensemble_coefs[2]}")
    #
    # # Try it out
    # tuned_lasso_pipe.fit(X_train, y_train)
    # tuned_rfr_pipe.fit(X_train, y_train)
    # svr_pipe.fit(X_train, y_train)
    # ensemble.fit(X_train, y_train)
    #
    # compare = pd.DataFrame(y_test.copy())
    # compare["LASSO"] = tuned_lasso_pipe.predict(X_test)
    # compare["RFR"] = tuned_rfr_pipe.predict(X_test)
    # compare["SVR"] = svr_pipe.predict(X_test)
    # compare["Ensemble"] = ensemble.predict(X_test)
    # compare = compare.apply(round)
    #
    # year = 2019
    # compare.loc[compare.index.get_level_values("season") == year]
    #
    # print("DONE")
    #
