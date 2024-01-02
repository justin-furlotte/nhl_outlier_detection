import pandas as pd
from utils.constants import dtypes


def load_data(fp: str) -> pd.DataFrame:

    df = pd.read_csv(fp, index_col=["playerId", "season"])
    for col in dtypes.keys():
        df[col] = df[col].apply(dtypes[col])
    df.index = pd.MultiIndex.from_tuples(
        [(int(player_id), int(season)) for player_id, season in df.index],
        names=df.index.names
    )
    return df


def get_id_name_map(df):
    id_name_map = {}
    for player_id in {x[0] for x in df.index}:
        name = df.loc[player_id, :]["name"].unique()[0]
        id_name_map[player_id] = name
    return id_name_map
