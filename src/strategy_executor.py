import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xgboost import XGBClassifier, XGBRegressor


class StrategyExecutor:
    def __init__(self, file_path, strategy):
        self.file_path = file_path
        self.strategy = strategy
        self.df = pd.read_csv(file_path)

    # --------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------

    def handle_missing(self):
        config = self.strategy.preprocessing.missing

        if config.strategy == "none":
            return

        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue

            if config.strategy == "mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())

            elif config.strategy == "median":
                self.df[col] = self.df[col].fillna(self.df[col].median())

            elif config.strategy == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def handle_encoding(self):
        config = self.strategy.preprocessing.encoding

        categorical_cols = self.df.select_dtypes(include="object").columns

        if config.method == "label":
            for col in categorical_cols:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))

        elif config.method == "one_hot":
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)

    def handle_scaling(self):
        config = self.strategy.preprocessing.scaling

        if config.method == "none":
            return

        numerical_cols = self.df.select_dtypes(include="number").columns

        if config.method == "standard":
            scaler = StandardScaler()
        elif config.method == "minmax":
            scaler = MinMaxScaler()
        else:
            return

        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])

    # --------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------

    def get_model(self, name, params):
        name = name.lower().strip()

        if name in ["logisticregression", "logistic_regression"]:
            return LogisticRegression(**params)

        if name in ["randomforestclassifier", "randomforest"]:
            return RandomForestClassifier(**params)

        if name in ["xgboost", "xgbclassifier"]:
            return XGBClassifier(**params)

        if name in ["linearregression", "linear_regression"]:
            return LinearRegression(**params)

        if name in ["randomforestregressor"]:
            return RandomForestRegressor(**params)

        if name in ["xgbregressor"]:
            return XGBRegressor(**params)

        raise ValueError(f"Unsupported model: {name}")

    # --------------------------------------------------
    # EXECUTION
    # --------------------------------------------------

    def execute(self):

        target_col = self.strategy.target_column

        self.handle_missing()
        self.handle_encoding()
        self.handle_scaling()

        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        split_config = self.strategy.modeling.split

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=split_config.test_size,
            random_state=split_config.random_state,
            stratify=y if split_config.stratified and self.strategy.task_type == "classification" else None
        )

        results = []

        for model_config in self.strategy.modeling.candidates:

            model = self.get_model(model_config.name, model_config.parameters)

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            if self.strategy.task_type == "classification":

                if self.strategy.modeling.evaluation_metric == "accuracy":
                    score = accuracy_score(y_test, preds)

                elif self.strategy.modeling.evaluation_metric == "f1":
                    score = f1_score(y_test, preds)

                else:
                    score = accuracy_score(y_test, preds)

            else:
                mse = mean_squared_error(y_test, preds)
                score = np.sqrt(mse)

            results.append({
                "model": model_config.name,
                "score": score
            })

        return results  