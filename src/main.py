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


if __name__ == "__main__":

    # Beginning and end of the training data (inclusive). Note that "2010" represents the 2010-2011 season, for example.
    train_start = 2010
    train_end = 2019

    target = "I_F_goals"  # "I_F_points", "I_F_primaryAssists", "I_F_secondaryAssists"
    situation = "all"  # "5on5", "4on5", "5on4"

    base_fp = "../data"
    fp = f"{base_fp}/Players_All_Years.csv"
    df = pd.read_csv(fp, index_col=["playerId", "season"])
    for col in dtypes.keys():
        df[col] = df[col].apply(dtypes[col])
    df = df.loc[df["situation"] == situation]
    df.index = pd.MultiIndex.from_tuples(
        [(int(player_id), int(season)) for player_id, season in df.index],
        names=df.index.names
    )

    id_name_map = {}
    for player_id in {x[0] for x in df.index}:
        name = df.loc[player_id, :]["name"].unique()[0]
        id_name_map[player_id] = name

    train_seasons = [train_start + i for i in range(train_end - train_start + 1)]
    df_train = df.loc[df.index.get_level_values("season").isin(train_seasons)]
    df_test = df.loc[df.index.get_level_values("season") == train_end + 1]

    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]

    cleaner = Preprocessor()
    preprocessor = cleaner.create_preprocessor(X_train=X_train)
    new_feature_names = cleaner.get_new_feature_names(X=X_train)

    tuner = Tuner(preprocessor=preprocessor, X_train=X_train, y_train=y_train)

    # ---------------- Create a pipeline for some models ----------------

    # Already trained a bunch of models and pickled them; let's load them
    with open(f"../models/LASSO_pickle_{train_start}_{train_end}", "rb") as f:
        tuned_lasso_pipe = pickle.load(f)
    with open(f"../models/RFR_pickle_{train_start}_{train_end}", "rb") as f:
        tuned_rfr_pipe = pickle.load(f)
    with open(f"../models/SVR_pickle_{train_start}_{train_end}", "rb") as f:
        svr_pipe = pickle.load(f)
    with open(f"../models/SR_pickle_{train_start}_{train_end}", "rb") as f:
        ensemble = pickle.load(f)

    # # LASSO Regression
    # lasso_pipe = make_pipeline(preprocessor, StandardScaler(), Lasso(max_iter=1000))
    # tuned_lasso_pipe = tuner.tune_lasso(lasso_pipe=lasso_pipe, verbose=True)
    #
    # # Random forest regression
    # rfr_pipe = make_pipeline(preprocessor, RandomForestRegressor())
    # tuned_rfr_pipe = tuner.tune_random_forest_regressor(rfr_pipe=rfr_pipe, verbose=True)
    #
    # # SVM Regression
    # # Unfortunately it is too slow to do hyperparameter optimization on SVR, but it is still giving
    # # decent results with a regularization strength of 1 and Gaussian RBF kernel
    # svr_pipe = make_pipeline(preprocessor, StandardScaler(), SVR(kernel="rbf", C=1.0))
    #
    # # Create and tune a stacking ensemble model with Ridge as the final regressor.
    # ensemble = tuner.tune_ensemble(pipe_lasso=tuned_lasso_pipe, pipe_rfr=tuned_rfr_pipe, pipe_svr=svr_pipe, verbose=True)
    #
    # # save models
    # print("Saving models...\n")
    # with open(f"../models/LASSO_pickle_{train_start}_{train_end}", "wb") as f:
    #     pickle.dump(tuned_lasso_pipe, f)
    # with open(f"../models/RFR_pickle_{train_start}_{train_end}", "wb") as f:
    #     pickle.dump(tuned_rfr_pipe, f)
    # with open(f"../models/SVR_pickle_{train_start}_{train_end}", "wb") as f:
    #     pickle.dump(svr_pipe, f)
    # with open(f"../models/SR_pickle_{train_start}_{train_end}", "wb") as f:
    #     pickle.dump(ensemble, f)

    ensemble_coefs = ensemble.final_estimator_.coef_
    print(f"Stacking estimator (Ridge) coefficients:")
    print(f"LASSO: {ensemble_coefs[0]}")
    print(f"RFR: {ensemble_coefs[1]}")
    print(f"SVR: {ensemble_coefs[2]}")

    # Try it out
    tuned_lasso_pipe.fit(X_train, y_train)
    tuned_rfr_pipe.fit(X_train, y_train)
    svr_pipe.fit(X_train, y_train)
    ensemble.fit(X_train, y_train)

    compare = pd.DataFrame(y_test.copy())
    compare["LASSO"] = tuned_lasso_pipe.predict(X_test)
    compare["RFR"] = tuned_rfr_pipe.predict(X_test)
    compare["SVR"] = svr_pipe.predict(X_test)
    compare["Ensemble"] = ensemble.predict(X_test)
    compare = compare.apply(round)

    year = 2019
    compare.loc[compare.index.get_level_values("season") == year]

    print("DONE")

