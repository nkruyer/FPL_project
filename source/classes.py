from source.ML_analysis import *

# from source.ML_analysis import build_svr
# from source.ML_analysis import build_svc
# from source.ML_analysis import build_knn
# from source.ML_analysis import build_knn_clf
# from source.ML_analysis import build_dtr
# from source.ML_analysis import build_mlp
# from source.ML_analysis import prepare_model_input

from sklearn.metrics import r2_score
from source.FPL_data_pull import *

# from source.FPL_data_pull import FPL_pull
# from source.FPL_data_pull import fetch_gameweek_data
# from source.FPL_data_pull import parse_teams
# from source.FPL_data_pull import parse_players
# from source.FPL_data_pull import add_classifier
import os
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
import pandas as pd


class FPL_data:
    def __init__(self, input_df=None):
        # eventually add functions from FPL data_pull here so this can populate automatically
        if input_df == None:
            self.teams, self.players, self.fixtures = FPL_pull()
            # self.team_dict = parse_teams(self.teams)
            # self.player_dict = parse_players(self.players)
            self.input_data, self.team_dict, self.player_dict = fetch_gameweek_data(
                self.players, self.teams
            )
            self.ML_input = prepare_model_input(self.input_data)
            self.ML_input_classifier = add_classifier(self.ML_input)
        else:
            self.teams, self.players, self.fixtures = FPL_pull()
            self.input_data = input_df
            self.team_dict = parse_teams(self.teams)
            self.player_dict = parse_players(self.players)
            self.ML_input = prepare_model_input(self.input_data)
            self.ML_input_classifier = add_classifier(self.ML_input)
        # should add all attributes as None during initilization for better code readability

    def build_svr(
        self,
        variable="total_points",
        type_scaler="standard",
        kernel_type="rbf",
        feature_selection=r_regression,
    ):
        (
            self.svr_regression,
            self.scaler,
            self.feature_selector,
            self.label_encoder,
            self.train_data_type,
            self.label_data_type,
            self.data_dict,
        ) = build_svr(
            self.ML_input,
            predicted_value=variable,
            scaler_type=type_scaler,
            kernel_type=kernel_type,
            feature_selection=feature_selection,
        )

    def build_svc(
        self,
        classifier="classifier",
        type_scaler="standard",
        kernel_type="rbf",
        feature_selection=r_regression,
    ):
        (
            self.svc_classifier,
            self.scaler,
            self.feature_selector,
            self.label_encoder,
            self.train_data_type,
            self.label_data_type,
            self.data_dict,
        ) = build_svc(
            self.ML_input_classifier,
            classifier=classifier,
            scaler_type=type_scaler,
            kernel_type=kernel_type,
            feature_selection=feature_selection,
        )

    def build_knn(
        self,
        variable="total_points",
        type_scaler="standard",
        weights="uniform",
        n_neighbors=5,
        feature_selection=r_regression,
    ):
        (
            self.knn_regression,
            self.scaler,
            self.feature_selector,
            self.label_encoder,
            self.train_data_type,
            self.label_data_type,
            self.data_dict,
        ) = build_knn(
            self.ML_input,
            predicted_value=variable,
            scaler_type=type_scaler,
            weights=weights,
            n_neighbors=n_neighbors,
            feature_selection=feature_selection,
        )

    def build_knn_clf(
        self,
        classifier="classifier",
        type_scaler="standard",
        weights="uniform",
        n_neighbors=5,
        feature_selection=r_regression,
    ):
        # clf, scaler, feature_selector, label_encoder, d_type, y_type, data
        (
            self.knn_classifier,
            self.scaler,
            self.feature_selector,
            self.label_encoder,
            self.train_data_type,
            self.label_data_type,
            self.data_dict,
        ) = build_knn_clf(
            self.ML_input_classifier,
            classifier=classifier,
            scaler_type=type_scaler,
            weights=weights,
            n_neighbors=n_neighbors,
            feature_selection=feature_selection,
        )

    def build_dtr(
        self,
        variable="total_points",
        type_scaler="standard",
        criterion="squared_error",
        feature_selection=r_regression,
    ):
        (
            self.dtr_regression,
            self.scaler,
            self.feature_selector,
            self.label_encoder,
            self.train_data_type,
            self.label_data_type,
            self.data_dict,
        ) = build_dtr(
            self.ML_input,
            predicted_value=variable,
            scaler_type=type_scaler,
            criterion=criterion,
            feature_selection=feature_selection,
        )

    def build_dtc(
        self,
        classifier="classifier",
        type_scaler="standard",
        criterion="squared_error",
        feature_selection=r_regression,
    ):
        (
            self.dtc_classifier,
            self.scaler,
            self.feature_selector,
            self.label_encoder,
            self.train_data_type,
            self.label_data_type,
            self.data_dict,
        ) = build_dtc(
            self.ML_input_classifier,
            classifier=classifier,
            scaler_type=type_scaler,
            criterion=criterion,
            feature_selection=feature_selection,
        )

    def build_mlp(
        self,
        variable="total_points",
        type_scaler="standard",
        max_iter=500,
        feature_selection=r_regression,
    ):
        (
            self.mlp_regression,
            self.scaler,
            self.feature_selector,
            self.label_encoder,
            self.train_data_type,
            self.label_data_type,
            self.data_dict,
        ) = build_mlp(
            self.ML_input,
            predicted_value=variable,
            scaler_type=type_scaler,
            max_iter=max_iter,
            feature_selection=feature_selection,
        )

    def build_mlp_clf(
        self,
        classifier="classifier",
        type_scaler="standard",
        max_iter=500,
        feature_selection=r_regression,
    ):
        (
            self.mlp_classifier,
            self.scaler,
            self.feature_selector,
            self.label_encoder,
            self.train_data_type,
            self.label_data_type,
            self.data_dict,
        ) = build_mlp_clf(
            self.ML_input_classifier,
            classifier=classifier,
            scaler_type=type_scaler,
            max_iter=max_iter,
            feature_selection=feature_selection,
        )

    def predict(self, model):
        X_test = self.data_dict["X_test"][self.train_data_type]
        y_test = self.data_dict["y_test"][self.label_data_type]

        self.data_dict["y_predicted"] = {}

        if model == "svr":
            y_predicted = self.svr_regression.predict(X_test)
            score = self.svr_regression.score(X_test, y_test)
        elif model == "knn":
            y_predicted = self.knn_regression.predict(X_test)
            score = self.knn_regression.score(X_test, y_test)
        elif model == "dtr":
            y_predicted = self.dtr_regression.predict(X_test)
            score = self.dtr_regression.score(X_test, y_test)
        elif model == "mlp":
            y_predicted = self.mlp_regression.predict(X_test)
            score = self.mlp_regression.score(X_test, y_test)
        elif model == "svc":
            y_predicted = self.svc_classifier.predict(X_test)
            score = self.svc_classifier.score(X_test, y_test)
        elif model == "knn_clf":
            y_predicted = self.knn_classifier.predict(X_test)
            score = self.knn_classifier.score(X_test, y_test)
        elif model == "dtc":
            y_predicted = self.dtc_classifier.predict(X_test)
            score = self.dtc_classifier.score(X_test, y_test)
        elif model == "mlp_clf":
            y_predicted = self.mlp_classifier.predict(X_test)
            score = self.mlp_classifier.score(X_test, y_test)

        self.data_dict["y_predicted"][model] = y_predicted
        self.model_score = score

        print(self.model_score)

    def retrain(self):
        # add info on repulling data and retraining
        pass

    def export_data(self):
        outdir = "/Users/nick/PycharmProjects/FPL_project/data"
        final_week = str(max(self.input_data["round"]))
        fname = f"performance_data_GW{final_week}.csv"
        outfile = os.path.join(outdir, fname)
        self.input_data.to_csv(outfile, index=False)

        print(f"Data saved as {outfile}")
