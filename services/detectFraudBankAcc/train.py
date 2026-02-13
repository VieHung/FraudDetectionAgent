import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from schema import RAW_FEATURES, TEST_PREDICT_OUTPUT_PATH

class Trainer:
    def __init__(self, data_path, model_dir):
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.data_path)

        self.X = df.drop(columns=["fraud_bool"])
        self.y = df["fraud_bool"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

    def evaluate(self, model, model_name):
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        pr_auc = average_precision_score(self.y_test, y_pred_proba)

        threshold, precision, recall = self.find_optimal_threshold(
            self.y_test, y_pred_proba, beta=2
        )

        print(f"\nModel: {model_name}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"Optimal threshold: {threshold:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall at that threshold: {recall:.4f}")

        return {
            "model": model_name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": threshold,
            "precision": precision,
            "recall": recall
        }

    def find_optimal_threshold(self, y_true, y_proba, beta):
        # (F2 score) Recall = beta^2 * Precision
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        precision = precision[:-1]
        recall = recall[:-1]

        fbeta = (1 + beta**2) * (precision * recall) / (
            (beta**2 * precision) + recall + 1e-9
        )

        best_idx = np.argmax(fbeta)

        return (
            thresholds[best_idx],
            precision[best_idx],
            recall[best_idx]
        )

    def save_model(self, model, model_name, metrics):
        path = os.path.join(self.model_dir, f"{model_name}.pkl")
        
        joblib.dump({
            "model": model,
            "features": list(self.X_train.columns),
            "raw_features": RAW_FEATURES,
            "threshold": metrics["threshold"],
            "metrics": metrics
        }, path)

        
        print(f"Saved {model_name} to {path}")

    # --- Train models

    def train_decision_tree(self):
        model = DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=42
        )

        model.fit(self.X_train, self.y_train)
        metrics = self.evaluate(model, "DecisionTree")
        self.save_model(model, "decision_tree", metrics)


    def train_random_forest(self):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)
        metrics = self.evaluate(model, "RandomForest")
        self.save_model(model, "random_forest", metrics)


    def train_xgboost(self):
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=self.get_scale_pos_weight(),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)
        metrics = self.evaluate(model, "XGBoost")
        self.save_model(model, "xgboost", metrics)


    def train_lightgbm(self):
        scale = self.get_scale_pos_weight()

        model = LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            random_state=42,
            scale_pos_weight=scale,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)
        metrics = self.evaluate(model, "LightGBM")
        self.save_model(model, "lightgbm", metrics)


    def get_scale_pos_weight(self):
        negative = (self.y_train == 0).sum()
        positive = (self.y_train == 1).sum()

        if positive == 0:
            return 1

        return negative / positive


    def train_all(self):
        self.train_decision_tree()
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
    
    # --- Evaluate
    def evaluate_all(self, model_dir):
        results = []

        for file in os.listdir(model_dir):
            if file.endswith(".pkl"):
                model_path = os.path.join(model_dir, file)
                model_name = file.replace(".pkl", "")

                print(f"\nLoading model: {model_name}")
                bundle = joblib.load(model_path)

                model = bundle["model"]
                threshold = bundle["threshold"]
                metrics = bundle["metrics"]

                results.append({
                    "model": model_name,
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "threshold": threshold,
                    "recall": metrics["recall"]
                })

        # Sort by PR-AUC
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="pr_auc", ascending=False)

        metrics_path = os.path.join(model_dir, "model_metrics.csv")
        results_df.to_csv(metrics_path, index=False)

        return results_df

    # --- Predict 
    def prepare_model_input(self, model_features, raw_features, raw_input: dict):
        df_input = pd.DataFrame([raw_input])

        for col in raw_features:
            if col not in df_input.columns:
                df_input[col] = -1

        df_input["missing_count"] = (df_input == -1).sum(axis=1)

        # Replace missing
        df_input["bank_months_count"] = df_input["bank_months_count"].replace(-1, 0)

        # Ratios
        df_input["credit_income_ratio"] = (
            df_input["proposed_credit_limit"] /
            (df_input["income"] + 1)
        )

        df_input["credit_per_account_month"] = (
            df_input["proposed_credit_limit"] /
            (df_input["bank_months_count"] + 1)
        )

        df_input["risk_age_ratio"] = (
            df_input["credit_risk_score"] /
            (df_input["customer_age"] + 1)
        )

        df_input["velocity_per_session"] = (
            df_input["velocity_6h"] /
            (df_input["session_length_in_minutes"] + 1)
        )

        df_input["low_name_email_similarity"] = (
            df_input["name_email_similarity"] < 0.3
        ).astype(int)

        df_input["cross_email_risk"] = (
            df_input["date_of_birth_distinct_emails_4w"] *
            df_input["device_distinct_emails_8w"]
        )

        df_input["velocity_spike"] = (
            df_input["velocity_6h"] /
            (df_input["velocity_4w"] + 1)
        )

        df_input["foreign_velocity"] = (
            df_input["foreign_request"] *
            df_input["velocity_6h"]
        )

        df_input["missing_credit_risk"] = (
            df_input["missing_count"] *
            df_input["proposed_credit_limit"]
        )

        categorical_cols = [
            "payment_type",
            "employment_status",
            "housing_status",
            "source",
            "device_os"
        ]

        df_input = pd.get_dummies(
            df_input,
            columns=[c for c in categorical_cols if c in df_input.columns],
            drop_first=False,
            dtype="int8"
        )

        df_input = df_input.replace([np.inf, -np.inf], np.nan)
        df_input = df_input.fillna(0)

        for col in model_features:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[model_features]

        return df_input

    def predict(self, model_name, raw_input):
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")

        bundle = joblib.load(model_path)

        model = bundle["model"]
        threshold = bundle["threshold"]
        model_features = bundle["features"]
        raw_features = bundle["raw_features"]

        print(f"Threshold used: {threshold:.4f}")

        X_input = self.prepare_model_input(model_features, raw_features, raw_input)

        proba = model.predict_proba(X_input)[:, 1]
        preds = (proba >= threshold).astype(int)

        return {
            "probability": float(proba[0]),
            "prediction": int(preds[0])
        }
    
    def predict_batch(self, model_name, input_csv_path):
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"{input_csv_path} not found")

        df_raw = pd.read_csv(input_csv_path)

        if "fraud_bool" in df_raw.columns:
            df_raw = df_raw.drop(columns=["fraud_bool"])

        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        bundle = joblib.load(model_path)

        model = bundle["model"]
        threshold = bundle["threshold"]
        model_features = bundle["features"]
        raw_features = bundle["raw_features"]

        processed_rows = []

        for _, row in df_raw.iterrows():
            processed = self.prepare_model_input(
                model_features,
                raw_features,
                row.to_dict()
            )
            processed_rows.append(processed)

        X_all = pd.concat(processed_rows, ignore_index=True)

        probabilities = model.predict_proba(X_all)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        df_raw["fraud_probability"] = probabilities
        df_raw["fraud_prediction"] = predictions

        df_raw.to_csv(TEST_PREDICT_OUTPUT_PATH, index=False)

        print(f"Predictions saved to {TEST_PREDICT_OUTPUT_PATH}")



