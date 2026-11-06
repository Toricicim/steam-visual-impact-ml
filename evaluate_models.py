"""Reproduce the original split and select alternatives using training-only CV."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


SEED = 42
ORIGINAL_FEATURES = ["Price", "Visual_Density", "Structure_Ratio", "Has_Demo"]
NUMERIC_FEATURES = ORIGINAL_FEATURES + ["Gif_Count", "Image_Count", "Bullet_Points", "Word_Count"]
CLASS_NAMES = ["0-49 reviews", "50-999 reviews", "1000+ reviews"]


def review_classes(counts):
    return np.where(counts < 50, 0, np.where(counts < 1000, 1, 2))


class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    """Fit two cumulative thresholds and project their probabilities to order."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1, 2]):
            raise ValueError("Expected all three ordered classes")
        self.models_ = [clone(self.estimator).fit(X, np.asarray(y) > threshold) for threshold in (0, 1)]
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        lower, upper = [model.predict_proba(X)[:, 1] for model in self.models_]
        # Project crossing cumulative probabilities to their common mean.
        crossing = lower < upper
        mean = (lower + upper) / 2
        lower = np.where(crossing, mean, lower)
        upper = np.where(crossing, mean, upper)
        return np.column_stack([1 - lower, lower - upper, upper])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def log_review_accuracy(y_true, y_pred):
    return accuracy_score(log_review_classes(y_true), log_review_classes(y_pred))


def log_review_balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(log_review_classes(y_true), log_review_classes(y_pred))


def log_review_macro_f1(y_true, y_pred):
    return f1_score(log_review_classes(y_true), log_review_classes(y_pred), average="macro")


def log_review_classes(values):
    # Compare in log space: expm1(log1p(50)) can round just below 50.
    return np.where(values < np.log1p(50), 0, np.where(values < np.log1p(1000), 1, 2))


def prepare_data(df):
    if df.AppID.duplicated().any():
        raise ValueError("Duplicate AppIDs would compromise the row-level split")
    # This allowlist excludes targets, review ratios and application identifiers.
    X = df[["Price", "Gif_Count", "Image_Count", "Bullet_Points", "Word_Count", "Has_Demo", "Tags", "Title"]].copy()
    X["Visual_Density"] = (2 * X.Gif_Count + X.Image_Count) / (X.Word_Count + 1)
    X["Structure_Ratio"] = X.Bullet_Points / (X.Word_Count + 1)
    y = review_classes(df.Review_Count)
    return X, y


def pipeline(model, features, *, tags=False, scale=False, title=None):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps += [("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]
    transformers = [("numeric", Pipeline(numeric_steps), features)]
    if tags:
        # Vocabulary is fitted inside each training fold, not on the whole CSV.
        transformers.append(("tags", CountVectorizer(token_pattern=r"[^|]+", binary=True, min_df=3), "Tags"))
    if title:
        vectorizer = TfidfVectorizer(analyzer=title, ngram_range=(1, 2) if title == "word" else (3, 5),
                                    min_df=2, max_features=2000, sublinear_tf=True)
        transformers.append(("title", vectorizer, "Title"))
    return Pipeline([
        ("features", ColumnTransformer(transformers, sparse_threshold=0)),
        ("model", model),
    ])


def candidates():
    models = {
        "majority_baseline": pipeline(DummyClassifier(strategy="most_frequent"), ORIGINAL_FEATURES),
        "original_rf": pipeline(RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=1), ORIGINAL_FEATURES),
        "regularized_rf_original_features": pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=1), ORIGINAL_FEATURES),
        "logistic_numeric": pipeline(LogisticRegression(C=1, max_iter=2000, random_state=SEED), NUMERIC_FEATURES, scale=True),
    }
    for leaf in (1, 4, 10):
        models[f"rf_numeric_leaf{leaf}"] = pipeline(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, max_features=1.0, random_state=SEED, n_jobs=1), NUMERIC_FEATURES)
    for leaf in (1, 4):
        models[f"extra_trees_numeric_leaf{leaf}"] = pipeline(
            ExtraTreesClassifier(n_estimators=300, min_samples_leaf=leaf, random_state=SEED, n_jobs=1), NUMERIC_FEATURES)
    for nodes in (7, 15, 31):
        models[f"hist_boost_numeric_nodes{nodes}"] = pipeline(
            HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=nodes, l2_regularization=1, early_stopping=True, random_state=SEED), NUMERIC_FEATURES)
    for strength in (0.3, 1.0):
        models[f"logistic_numeric_tags_C{strength}"] = pipeline(
            LogisticRegression(C=strength, max_iter=2000, random_state=SEED), NUMERIC_FEATURES, tags=True, scale=True)
    models["rf_numeric_tags"] = pipeline(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=1), NUMERIC_FEATURES, tags=True)
    # Fixed second-stage candidates: deeper approaches, still without new data.
    models["balanced_logistic_numeric_tags"] = pipeline(
        LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=SEED), NUMERIC_FEATURES, tags=True, scale=True)
    for analyzer in ("word", "char_wb"):
        models[f"logistic_numeric_tags_title_{analyzer}"] = pipeline(
            LogisticRegression(C=1, max_iter=2000, random_state=SEED), NUMERIC_FEATURES, tags=True, scale=True, title=analyzer)
    models["ordinal_logistic_numeric_tags"] = pipeline(
        OrdinalClassifier(LogisticRegression(C=1, max_iter=2000, random_state=SEED)), NUMERIC_FEATURES, tags=True, scale=True)
    for tags in (False, True):
        suffix = "numeric_tags" if tags else "numeric"
        models[f"ordinal_hist_boost_{suffix}"] = pipeline(
            OrdinalClassifier(HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, l2_regularization=1,
                                                           early_stopping=True, random_state=SEED)), NUMERIC_FEATURES, tags=tags)
        models[f"log_reviews_hist_boost_{suffix}"] = pipeline(
            HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, l2_regularization=1,
                                          early_stopping=True, random_state=SEED), NUMERIC_FEATURES, tags=tags)
    models["log_reviews_extra_trees_numeric_tags"] = pipeline(
        ExtraTreesRegressor(n_estimators=300, min_samples_leaf=4, random_state=SEED, n_jobs=1), NUMERIC_FEATURES, tags=True)
    for nodes in (7, 15):
        models[f"hist_boost_numeric_tags_nodes{nodes}"] = pipeline(
            HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=nodes, l2_regularization=1,
                                           early_stopping=True, random_state=SEED), NUMERIC_FEATURES, tags=True)
    return models


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }


def evaluate(data_path):
    df = pd.read_csv(data_path)
    X, y = prepare_data(df)
    # Preserve the notebook's original, unstratified 80/20 split for comparison.
    train, test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X.iloc[train], y[train]))
    models = candidates()
    results = []
    for name, model in models.items():
        count_regression = name.startswith("log_reviews_")
        fit_target = np.log1p(df.Review_Count.to_numpy()) if count_regression else y
        scoring = ({"accuracy": make_scorer(log_review_accuracy), "balanced_accuracy": make_scorer(log_review_balanced_accuracy),
                    "macro_f1": make_scorer(log_review_macro_f1)} if count_regression else
                   {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"})
        scores = cross_validate(model, X.iloc[train], fit_target[train], cv=cv, n_jobs=1, scoring=scoring, error_score="raise")
        result = {"name": name, "stage": 1 if len(results) < 15 else 2,
                  "parameters": {key: value for key, value in model.get_params().items()
                                 if key.startswith("model__") and isinstance(value, (str, int, float, bool, type(None)))}}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = scores[f"test_{metric}"]
            result[f"cv_{metric}_mean"] = float(values.mean())
            result[f"cv_{metric}_std"] = float(values.std(ddof=1))
            result[f"cv_{metric}_folds"] = values.tolist()
        results.append(result)
        print(f"{name}: CV accuracy={result['cv_accuracy_mean']:.4f}, macro F1={result['cv_macro_f1_mean']:.4f}", flush=True)

    # Fix the choice before accessing any candidate's held-out predictions.
    winner = max(results, key=lambda result: result["cv_accuracy_mean"])["name"]
    numeric_winner = max((result for result in results if "tags" not in result["name"]),
                         key=lambda result: result["cv_accuracy_mean"])["name"]
    macro_winner = max(results, key=lambda result: result["cv_macro_f1_mean"])["name"]
    heldout = {}
    predictions = {}
    for name in dict.fromkeys(["majority_baseline", "original_rf", numeric_winner, winner, macro_winner,
                              "rf_numeric_tags", "logistic_numeric_tags_C1.0"]):
        count_regression = name.startswith("log_reviews_")
        fit_target = np.log1p(df.Review_Count.to_numpy()) if count_regression else y
        fitted = clone(models[name]).fit(X.iloc[train], fit_target[train])
        predicted = fitted.predict(X.iloc[test])
        if count_regression:
            predicted = log_review_classes(predicted)
        predictions[name] = predicted
        heldout[name] = metrics(y[test], predicted)
        print(f"{name}: held-out accuracy={heldout[name]['accuracy']:.4f}", flush=True)

    # Paired resampling describes uncertainty on this historical test sample.
    rng = np.random.default_rng(SEED)
    original_correct = predictions["original_rf"] == y[test]
    paired = {}
    for name in dict.fromkeys([numeric_winner, winner, macro_winner]):
        difference = (predictions[name] == y[test]).astype(float) - original_correct.astype(float)
        samples = rng.integers(len(test), size=(5000, len(test)))
        low, high = np.quantile(difference[samples].mean(axis=1), [0.025, 0.975])
        paired[name] = {"accuracy_delta": float(difference.mean()), "bootstrap_95pct_interval": [float(low), float(high)]}

    return {
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "rows": len(df), "unique_appids": int(df.AppID.nunique()),
        "zero_review_rows": int((df.Review_Count == 0).sum()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "split": {"seed": SEED, "train_size": len(train), "test_size": len(test), "stratified_holdout": False, "test_appids": df.iloc[test].AppID.tolist()},
        "selection": "Highest mean accuracy in 5-fold stratified CV on training rows only; a numeric-only winner and a secondary macro-F1 winner are also reported. Candidate identities follow CV scores, not held-out rankings.",
        "experiment_stages": "Stage 1: 15 baseline/classification candidates. Stage 2: 11 prelisted candidates using ordered thresholds, direct log-review regression, title text or boosted tag features, added after inspecting stage 1. All CV fits exclude historical test rows.",
        "numeric_features": NUMERIC_FEATURES,
        "tag_features": "Pipe-separated store tags, fold-local binary vocabulary with min_df=3; broader metadata experiment.",
        "title_features": "Second-stage text experiments: fold-local word or character TF-IDF, min_df=2, at most 2000 terms; broader metadata experiment.",
        "selected_model": winner,
        "selected_numeric_model": numeric_winner,
        "selected_macro_f1_model": macro_winner,
        "cross_validation": results,
        "heldout": heldout,
        "paired_accuracy_comparison_to_original": paired,
        "limitations": [
            "The original holdout was already used by the historical notebook; this is a development comparison, not a fresh external validation.",
            "The secondary macro-F1 comparison was added after observing the accuracy-selected model's weak middle-class recall; interpret it as exploratory.",
            "Second-stage methods were motivated by stage-one findings. Reusing CV folds and the historical holdout makes this adaptive development, not an independent confirmatory study.",
            "Selecting among multiple CV results can make the winning CV score optimistic; external or nested validation is still needed.",
            "Review-count categories are proxies, not measured sales or profit.",
            "Store-page snapshots and tags were not verified as available before launch; reverse causality and temporal leakage remain possible.",
            "Bootstrap intervals describe this sample and do not account for sampling bias or model selection.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("steam_indie_games_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reports" / "model_comparison.json")
    args = parser.parse_args()
    report = evaluate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
