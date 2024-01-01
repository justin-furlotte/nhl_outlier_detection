from utils.constants import dtypes
from utils.preprocessing import Preprocessor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from sklearn.pipeline import Pipeline, make_pipeline

import pandas as pd
import numpy as np
pd.set_option('expand_frame_repr', False)  # Display full dataframes while printing them


def create_df(dfs):
    for key in dfs.keys():
        # tonjt = self.dfs[key].columns
        dfs[key] = dfs[key].loc[dfs[key]["situation"] == "all"].drop(columns="situation")
    return dfs
def create_x_y_train(self, season_start, season_end):

    X_all_years = [self.dfs[key].drop(columns="I_F_goals") for key in self.dfs.keys()]
    y_all_years = [self.dfs[key]["I_F_goals"] for key in self.dfs.keys()]

    start = int(season_start[:2])
    finish = int(season_end[:2])

    X = []
    y = []

    for i in np.arange(start, finish + 1):
        dfX = X_all_years[i - start]

        playerid_new = {playerid: str(playerid) + "_" + str(i) + "_" + str(i + 1) for playerid in list(dfX.index)}
        dfX = dfX.rename(index=playerid_new)
        X.append(dfX)

        dfy = y_all_years[i - start]
        dfy = dfy.rename(index=playerid_new)
        y.append(dfy)

    return pd.concat(X), pd.concat(y)


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
    df_train = df.loc[df.index.get_level_values('season').isin(train_seasons)]

    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]

    cleaner = Preprocessor()
    preprocessor = cleaner.create_preprocessor(X_train=X_train)
    new_feature_names = cleaner.get_new_feature_names(X=X_train)

    pipe_lasso = make_pipeline(preprocessor, StandardScaler(), Lasso(max_iter=1000))

    print("DONE")

