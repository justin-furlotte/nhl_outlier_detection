import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:

    def __init__(self, num_features: list[str] = None, categorical_features: list[str] = None):

        if num_features is not None:
            self.num_features = num_features
        else:
            # Some carefully selected numerical features:
            self.num_features = [
                'games_played', 'icetime', 'gameScore', 'onIce_corsiPercentage',
                'onIce_fenwickPercentage', 'I_F_primaryAssists', 'I_F_secondaryAssists',
                'I_F_shotsOnGoal', 'I_F_missedShots', 'I_F_blockedShotAttempts', 'I_F_shotAttempts',
                'I_F_rebounds', 'I_F_freeze', 'I_F_playStopped', 'I_F_playContinuedInZone',
                'I_F_playContinuedOutsideZone', 'I_F_hits', 'I_F_takeaways', 'I_F_giveaways',
                'I_F_lowDangerShots', 'I_F_mediumDangerShots', 'I_F_highDangerShots',
                'I_F_scoreAdjustedShotsAttempts', 'I_F_unblockedShotAttempts', 'I_F_shifts',
                'I_F_oZoneShiftStarts', 'I_F_dZoneShiftStarts', 'I_F_neutralZoneShiftStarts',
                'I_F_flyShiftStarts', 'I_F_oZoneShiftEnds', 'I_F_dZoneShiftEnds',
                'I_F_neutralZoneShiftEnds', 'I_F_flyShiftEnds', 'faceoffsWon', 'faceoffsLost',
                'penalityMinutesDrawn', 'penaltiesDrawn', 'OnIce_F_shotsOnGoal', 'OnIce_F_missedShots',
                'OnIce_F_blockedShotAttempts', 'OnIce_F_shotAttempts', 'OnIce_F_rebounds',
                'OnIce_F_lowDangerShots', 'OnIce_F_mediumDangerShots', 'OnIce_F_highDangerShots',
                'OnIce_F_unblockedShotAttempts', 'corsiForAfterShifts',
                'corsiAgainstAfterShifts', 'fenwickForAfterShifts', 'fenwickAgainstAfterShifts'
            ]

        if categorical_features is not None:
            self.categorical_features = categorical_features
        else:
            self.categorical_features = ["position"]

        self.preprocessor = None
        self.transformed_feature_names = None

    def create_preprocessor(self, X_train: pd.DataFrame):

        # drop everything other than the numerical and categorical features
        drop_features = []
        for feat in X_train.columns:
            if feat not in self.num_features + self.categorical_features:
                drop_features.append(feat)

        pass_features = X_train.columns.tolist()
        for feat in drop_features + self.categorical_features:  # everything else is passed through
            pass_features.remove(feat)

        # Keep track of new feature names
        new_categorical_features = []
        for cat_feat in self.categorical_features:
            for feat_name in X_train[cat_feat].unique().tolist():
                new_categorical_features.append(feat_name)

        # Apply the following tranforms (in this order):
        #   1. One-hot encoder to the categorical features
        #   2. Standard Scaling to the numerical features
        #   3. Drop all other columns
        preprocessor = make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore", sparse=False), self.categorical_features),
            (StandardScaler(), self.num_features),
            ("drop", drop_features)
        )

        self.preprocessor = preprocessor

        # Note that we will apply standard scaler later in a pipeline
        return preprocessor

    def get_new_feature_names(self, X):

        _ = self.preprocessor.fit_transform(X)
        ohe_column_names = self.preprocessor.named_transformers_[
            "onehotencoder"].get_feature_names_out().tolist()  # [0:51]
        new_feature_names = ohe_column_names + self.num_features

        self.transformed_feature_names = new_feature_names

        return new_feature_names
