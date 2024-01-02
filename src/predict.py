from utils.constants import dtypes
from utils.preprocessing import Preprocessor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from utils.hyperparameter_tuning import Tuner
from utils.data_loading import load_data, get_id_name_map
import pandas as pd
import numpy as np
import pickle
pd.set_option('expand_frame_repr', False)  # Display full dataframes while printing them


def predict_from_pickled_models():

    predictions = []

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

                # Already trained a bunch of models with optimally selected hyperparameters and pickled them;
                # let's load them.
                with open(f"../models/LASSO_pickle_{target}_{situation}_{train_start}_{train_end}", "rb") as f:
                    tuned_lasso_pipe = pickle.load(f)
                with open(f"../models/RFR_pickle_{target}_{situation}_{train_start}_{train_end}", "rb") as f:
                    tuned_rfr_pipe = pickle.load(f)
                with open(f"../models/SVR_pickle_{target}_{situation}_{train_start}_{train_end}", "rb") as f:
                    svr_pipe = pickle.load(f)
                with open(f"../models/SR_pickle_{target}_{situation}_{train_start}_{train_end}", "rb") as f:
                    ensemble = pickle.load(f)

                # Load data
                df = load_data(fp=fp)

                id_name_map = get_id_name_map(df=df)

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

                # Train the model
                ensemble.fit(X_train, y_train)
                ensemble_pred = ensemble.predict(X_test)

                pred = X_test[["name", "team", "games_played"]]
                pred.loc[:, "Season"] = [x[1] for x in pred.index]
                pred = pred.merge(y_test, left_index=True, right_index=True)
                pred.loc[:, f"{target} Pace"] = round(pred[target] / pred["games_played"] * 82)
                pred.drop([target, "games_played"], axis=1, inplace=True)
                pred = pred.rename(columns={"name": "Player", "team": "Team"})

                pred["Predicted Goal Pace"] = ensemble_pred

                predictions.append(pred)

    prediction = pd.concat(predictions, axis=1)
    prediction.to_csv("../graphing_data/scatter_df_jan_1_2024.csv")
