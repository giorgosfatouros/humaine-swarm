"""
Smart Cities - Auto-Correction Eligibility Classifier (KFP v2 pipeline).

HumAIne WP6 Smart Cities pilot, built for the Swarm Assistant / Kubeflow demo.

When a citizen's entry on a parking-permit application differs from the correct
value, the mistake is either obvious (a sound-alike misspelling, a misplaced ZIP
code) and can be fixed automatically, or non-obvious (a different name, a wrong
number) and must go to a human operator. This pipeline trains two small
classifiers on the synthetic "obvious mistake" dataset to make that call.

Steps:
    load_data        read train.csv / test.csv from MinIO
    build_features   string-difference features per (supplied, correct) pair
    train_model      logistic regression and random forest
    evaluate_model   metrics, confusion matrix, HTML report and plot tables per model
    publish_results  upload everything to MinIO under
                     kubeflow/smart-cities-auto-correction-classifier/<run_id>/

The MinIO credential parameters are filled automatically when the pipeline is
launched from the HumAIne Swarm Assistant (the user's own STS session keys).

Compile:
    poetry run python kubeflow/smart_cities/auto_correction_pipeline.py
"""
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import HTML, Artifact, ClassificationMetrics, Dataset, Input, Metrics, Model, Output

PIPELINE_NAME = "smart-cities-auto-correction-classifier"
PIPELINE_DESCRIPTION = (
    "Smart Cities - Auto-Correction Eligibility Classifier: decides whether a "
    "citizen's mistaken application entry can be fixed automatically or needs a "
    "human operator. Trains a logistic regression and a random forest on the "
    "synthetic IBM2 'obvious mistake' dataset and publishes models, metrics and "
    "reports to MinIO."
)

BASE_IMAGE = "python:3.11-slim"
PANDAS = "pandas==2.2.3"
SKLEARN = "scikit-learn==1.5.2"
MINIO = "minio==7.2.15"
MATPLOTLIB = "matplotlib==3.9.2"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=[PANDAS, MINIO])
def load_data(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_session_token: str,
    data_bucket: str,
    train_object: str,
    test_object: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
):
    """Download the train and test CSVs from MinIO."""
    import pandas as pd
    from minio import Minio

    if not (minio_access_key and minio_secret_key):
        raise ValueError(
            "MinIO credentials are empty. Launch the pipeline from the HumAIne Swarm "
            "Assistant, which fills them from your session, or pass them explicitly."
        )
    secure = not minio_endpoint.startswith("http://")
    endpoint = minio_endpoint.split("://", 1)[-1].rstrip("/")
    client = Minio(
        endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        session_token=minio_session_token or None,
        secure=secure,
    )

    for obj, artifact in ((train_object, train_data), (test_object, test_data)):
        client.fget_object(data_bucket, obj, artifact.path)
        df = pd.read_csv(artifact.path, encoding="utf-8")
        artifact.metadata.update({
            "source": f"s3://{data_bucket}/{obj}",
            "rows": len(df),
            "auto_correctable": int(df["is_obvious"].sum()),
            "needs_operator": int((1 - df["is_obvious"]).sum()),
        })
        print(f"Loaded s3://{data_bucket}/{obj}: {len(df)} rows")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=[PANDAS])
def build_features(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    train_features: Output[Dataset],
    test_features: Output[Dataset],
):
    """Turn each (supplied value, correct value, field type) into numeric features.

    The features are generic string distances between the citizen's entry and the
    correct value. Features that restate the dataset's generation rules (Greek
    sound-alike spelling, token reordering) are left out on purpose: with them both
    models score 100% and the demo has nothing to compare. `mutation_type` names
    the rule that created the mistake and is never used.
    """
    import re
    import unicodedata
    from collections import Counter

    import pandas as pd

    field_types = ["name", "surname", "father_name", "address", "vat_number",
                   "id_number", "power_supply", "licence_plate", "mobile_phone"]
    numeric_fields = {"vat_number", "id_number", "power_supply", "licence_plate", "mobile_phone"}

    def clean(s):
        s = unicodedata.normalize("NFD", str(s).upper())
        return "".join(c for c in s if not unicodedata.combining(c)).strip()

    def levenshtein(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def featurize(df):
        rows = []
        for sup, gt, ft in zip(df["supplied_value"], df["ground_truth"], df["field_type"]):
            s, g = clean(sup), clean(gt)
            lev = levenshtein(s, g)
            row = {f"field_{f}": int(ft == f) for f in field_types}
            row.update({
                "is_numeric_field": int(ft in numeric_fields),
                "len_diff": len(s) - len(g),
                "levenshtein": lev,
                "levenshtein_norm": lev / max(len(s), len(g), 1),
                "char_multiset_equal": int(Counter(s.replace(" ", "")) == Counter(g.replace(" ", ""))),
                "digits_equal": int(re.sub(r"\D", "", s) == re.sub(r"\D", "", g)),
            })
            rows.append(row)
        out = pd.DataFrame(rows)
        out["field_type"] = df["field_type"].values
        out["label"] = df["is_obvious"].astype(int).values
        return out

    for src, dst in ((train_data, train_features), (test_data, test_features)):
        feats = featurize(pd.read_csv(src.path, encoding="utf-8"))
        feats.to_csv(dst.path, index=False)
        names = [c for c in feats.columns if c not in ("field_type", "label")]
        dst.metadata.update({"rows": len(feats), "n_features": len(names), "features": names})
    print(f"Built {len(names)} features: {names}")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=[PANDAS, SKLEARN])
def train_model(
    train_features: Input[Dataset],
    model_type: str,
    lr_c: float,
    rf_n_estimators: int,
    rf_max_depth: int,
    random_state: int,
    model: Output[Model],
):
    """Train a logistic regression or a random forest on the training features."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(train_features.path)
    feature_names = [c for c in df.columns if c not in ("field_type", "label")]
    X, y = df[feature_names], df["label"]

    if model_type == "logistic_regression":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=lr_c, max_iter=1000))
        params = {"C": lr_c}
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                     random_state=random_state)
        params = {"n_estimators": rf_n_estimators, "max_depth": rf_max_depth}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X, y)
    joblib.dump({"model": clf, "model_type": model_type, "feature_names": feature_names},
                model.path)
    model.metadata.update({"model_type": model_type, "train_rows": len(df), **params})
    print(f"Trained {model_type} on {len(df)} rows, train accuracy {clf.score(X, y):.4f}")


@dsl.component(base_image=BASE_IMAGE, packages_to_install=[PANDAS, SKLEARN, MATPLOTLIB])
def evaluate_model(
    model: Input[Model],
    test_features: Input[Dataset],
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    metrics_json: Output[Dataset],
    report: Output[HTML],
    plots: Output[Artifact],
):
    """Score the model on the held-out test set.

    Writes two kinds of HTML: `report` is a visual page (embedded PNGs) for people
    and the Kubeflow UI; `plots` is a folder of small text-only tables named
    <model>_{confusion_matrix,roc_curve,feature_importance}.html, which the
    assistant's get_pipeline_visualization tool reads and passes to the LLM.
    """
    import base64
    import io
    import json
    import os

    import joblib
    import matplotlib
    import pandas as pd
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score, roc_curve)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The model artifact was written by train_model in this same run, so unpickling is safe
    bundle = joblib.load(model.path)
    clf, model_type, feature_names = bundle["model"], bundle["model_type"], bundle["feature_names"]
    model_name = {"logistic_regression": "Logistic Regression",
                  "random_forest": "Random Forest"}.get(model_type, model_type)
    labels = ["needs operator", "auto-correctable"]

    df = pd.read_csv(test_features.path)
    X, y = df[feature_names], df["label"]
    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    per_field = {ft: round(float(accuracy_score(g["label"], y_pred[g.index])), 4)
                 for ft, g in df.groupby("field_type")}
    result = {
        "model_name": model_name,
        "model_type": model_type,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y, y_score)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "confusion_matrix_orientation": "rows = actual class, columns = predicted class",
        # Named counts, so a reader cannot swap actual and predicted. Wrong auto-corrections are the costly error.
        "outcomes": {
            "needs_operator_correctly_sent_to_operator": int(cm[0, 0]),
            "needs_operator_wrongly_auto_corrected": int(cm[0, 1]),
            "auto_correctable_wrongly_sent_to_operator": int(cm[1, 0]),
            "auto_correctable_correctly_auto_corrected": int(cm[1, 1]),
        },
        "positive_class": "auto-correctable",
        "per_field_accuracy": per_field,
        "test_rows": len(df),
    }
    with open(metrics_json.path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    for k in ("accuracy", "precision", "recall", "f1_score", "auc"):
        metrics.log_metric(k, round(result[k], 4))
    classification_metrics.log_confusion_matrix(labels, cm.tolist())

    # Confusion matrix and feature weights, embedded as PNGs in one HTML page
    if hasattr(clf, "feature_importances_"):
        weights, weight_title = clf.feature_importances_, "Feature importance"
    else:
        weights, weight_title = clf[-1].coef_[0], "Coefficient (standardised features)"
    order = sorted(range(len(feature_names)), key=lambda i: abs(weights[i]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    ax1.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax1.set_xticks([0, 1], labels)
    ax1.set_yticks([0, 1], labels)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion matrix (test set)")
    ax2.barh([feature_names[i] for i in order], [weights[i] for i in order], color="#3b6ea5")
    ax2.set_title(weight_title)
    fig.suptitle(f"{model_name} - auto-correction eligibility")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode("ascii")

    metric_rows = "".join(
        f"<tr><td>{k}</td><td>{result[k]:.4f}</td></tr>"
        for k in ("accuracy", "precision", "recall", "f1_score", "auc"))
    field_rows = "".join(f"<tr><td>{ft}</td><td>{acc:.4f}</td></tr>" for ft, acc in per_field.items())
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{model_name} - Auto-Correction Classifier</title>
<style>body{{font-family:sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;margin:8px 24px 16px 0;display:inline-table;vertical-align:top}}
td,th{{border:1px solid #ccc;padding:4px 10px;text-align:left}}th{{background:#f0f4f8}}img{{max-width:100%}}</style></head>
<body><h2>Smart Cities - Auto-Correction Eligibility Classifier: {model_name}</h2>
<p>Positive class: <b>auto-correctable</b> (obvious mistake). Negative class: <b>needs operator</b>. Test rows: {len(df)}.</p>
<table><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>
<table><tr><th>Field type</th><th>Accuracy</th></tr>{field_rows}</table>
<img src="data:image/png;base64,{img}" alt="Confusion matrix and feature weights"></body></html>"""
    with open(report.path, "w", encoding="utf-8") as fh:
        fh.write(html)

    # Compact table versions for the assistant (no images, a few hundred bytes each)
    def table_page(title, header, rows, note=""):
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        head = "".join(f"<th>{h}</th>" for h in header)
        note = f"<p>{note}</p>" if note else ""
        return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>'
                f"<body><h3>{title}</h3><table><tr>{head}</tr>{body}</table>{note}</body></html>")

    fpr, tpr, thr = roc_curve(y, y_score)
    step = max(1, len(fpr) // 20)
    roc_idx = sorted(set(range(0, len(fpr), step)) | {len(fpr) - 1})
    pages = {
        "confusion_matrix": table_page(
            f"{model_name} - confusion matrix (test set, rows = actual)",
            ["actual \\ predicted", *labels],
            [[labels[i], int(cm[i, 0]), int(cm[i, 1])] for i in range(2)],
            note=(f"Errors: {int(cm[0, 1])} entries that needed an operator would be wrongly "
                  f"auto-corrected (false positives, the costly error). {int(cm[1, 0])} "
                  f"auto-correctable entries would be sent to an operator unnecessarily "
                  f"(false negatives).")),
        "roc_curve": table_page(
            f"{model_name} - ROC curve (AUC {result['auc']:.4f})",
            ["threshold", "false_positive_rate", "true_positive_rate"],
            [[f"{min(thr[i], 1.0):.3f}", f"{fpr[i]:.3f}", f"{tpr[i]:.3f}"] for i in roc_idx]),
        "feature_importance": table_page(
            f"{model_name} - {weight_title.lower()}",
            ["feature", "weight"],
            [[feature_names[i], f"{weights[i]:.4f}"] for i in reversed(order)]),
    }
    os.makedirs(plots.path, exist_ok=True)
    for kind, page in pages.items():
        with open(os.path.join(plots.path, f"{model_type}_{kind}.html"), "w", encoding="utf-8") as fh:
            fh.write(page)

    print(f"{model_name}: " + ", ".join(f"{k}={result[k]:.4f}"
                                         for k in ("accuracy", "precision", "recall", "f1_score", "auc")))


@dsl.component(base_image=BASE_IMAGE, packages_to_install=[MINIO])
def publish_results(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_session_token: str,
    output_bucket: str,
    pipeline_name: str,
    run_id: str,
    data_bucket: str,
    train_object: str,
    test_object: str,
    lr_c: float,
    rf_n_estimators: int,
    rf_max_depth: int,
    random_state: int,
    lr_model: Input[Model],
    rf_model: Input[Model],
    lr_metrics: Input[Dataset],
    rf_metrics: Input[Dataset],
    lr_report: Input[HTML],
    rf_report: Input[HTML],
    lr_plots: Input[Artifact],
    rf_plots: Input[Artifact],
    summary: Output[Metrics],
):
    """Upload models, metrics and reports to MinIO and pick the best model by F1."""
    import datetime
    import io
    import json
    import os

    from minio import Minio

    secure = not minio_endpoint.startswith("http://")
    endpoint = minio_endpoint.split("://", 1)[-1].rstrip("/")
    client = Minio(
        endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        session_token=minio_session_token or None,
        secure=secure,
    )
    base = f"kubeflow/{pipeline_name}/{run_id}"

    uploads = [
        (lr_model.path, f"{base}/models/logistic_regression.joblib", "application/octet-stream"),
        (rf_model.path, f"{base}/models/random_forest.joblib", "application/octet-stream"),
        (lr_metrics.path, f"{base}/metrics/logistic_regression_metrics.json", "application/json"),
        (rf_metrics.path, f"{base}/metrics/random_forest_metrics.json", "application/json"),
        (lr_report.path, f"{base}/reports/logistic_regression_report.html", "text/html"),
        (rf_report.path, f"{base}/reports/random_forest_report.html", "text/html"),
    ]
    for plots in (lr_plots, rf_plots):
        for name in sorted(os.listdir(plots.path)):
            uploads.append((os.path.join(plots.path, name), f"{base}/plots/{name}", "text/html"))
    for src, dst, content_type in uploads:
        client.fput_object(output_bucket, dst, src, content_type=content_type)
        print(f"Uploaded s3://{output_bucket}/{dst}")

    results = []
    for path in (lr_metrics.path, rf_metrics.path):
        with open(path, encoding="utf-8") as fh:
            results.append(json.load(fh))
    best = max(results, key=lambda r: r["f1_score"])

    run_summary = {
        "pipeline_name": pipeline_name,
        "run_id": run_id,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": "Decide whether a citizen's mistaken entry can be auto-corrected "
                "(obvious mistake) or needs a human operator.",
        "data_source": {"train": f"s3://{data_bucket}/{train_object}",
                        "test": f"s3://{data_bucket}/{test_object}"},
        "hyperparameters": {"lr_c": lr_c, "rf_n_estimators": rf_n_estimators,
                            "rf_max_depth": rf_max_depth, "random_state": random_state},
        "best_model": best["model_name"],
        "selection_metric": "f1_score",
        "models": {r["model_name"]: {k: r[k] for k in ("accuracy", "precision", "recall",
                                                       "f1_score", "auc")}
                   for r in results},
        "artifacts_prefix": f"s3://{output_bucket}/{base}/",
    }
    # run_parameters.json is the file the assistant's compare/metrics tools read hyperparameters from
    run_parameters = {
        "pipeline_name": pipeline_name,
        "run_id": run_id,
        "parameters": {**run_summary["hyperparameters"],
                       "train_data": run_summary["data_source"]["train"],
                       "test_data": run_summary["data_source"]["test"]},
    }
    for name, doc in (("run_summary.json", run_summary), ("run_parameters.json", run_parameters)):
        body = json.dumps(doc, indent=2).encode("utf-8")
        client.put_object(output_bucket, f"{base}/metadata/{name}", io.BytesIO(body),
                          len(body), content_type="application/json")

    summary.log_metric("best_model_f1", round(best["f1_score"], 4))
    summary.metadata["best_model"] = best["model_name"]
    summary.metadata["artifacts_prefix"] = run_summary["artifacts_prefix"]
    print(f"Best model: {best['model_name']} (F1 {best['f1_score']:.4f})")


@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def auto_correction_pipeline(
    minio_endpoint: str = "s3-minio.humaine-horizon.eu",
    minio_access_key: str = "",
    minio_secret_key: str = "",
    minio_session_token: str = "",
    data_bucket: str = "smart-cities-data",
    train_object: str = "auto-correction/train.csv",
    test_object: str = "auto-correction/test.csv",
    output_bucket: str = "smart-cities-data",
    lr_c: float = 1.0,
    rf_n_estimators: int = 200,
    rf_max_depth: int = 8,
    random_state: int = 42,
):
    creds = dict(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_session_token=minio_session_token,
    )

    # No set_display_name(): the cluster's KFP driver records a task under its display name
    # but looks up upstream outputs by task name, and crashes (nil pointer) when they differ.
    data = load_data(data_bucket=data_bucket, train_object=train_object,
                     test_object=test_object, **creds)
    data.set_caching_options(False)  # credentials change per run

    features = build_features(train_data=data.outputs["train_data"],
                              test_data=data.outputs["test_data"])

    trained, evaluated = {}, {}
    for model_type in ("logistic_regression", "random_forest"):
        train = train_model(train_features=features.outputs["train_features"],
                            model_type=model_type, lr_c=lr_c, rf_n_estimators=rf_n_estimators,
                            rf_max_depth=rf_max_depth, random_state=random_state)
        evaluate = evaluate_model(model=train.outputs["model"],
                                  test_features=features.outputs["test_features"])
        trained[model_type], evaluated[model_type] = train, evaluate

    publish = publish_results(
        output_bucket=output_bucket,
        pipeline_name=PIPELINE_NAME,
        run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        data_bucket=data_bucket,
        train_object=train_object,
        test_object=test_object,
        lr_c=lr_c,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        random_state=random_state,
        lr_model=trained["logistic_regression"].outputs["model"],
        rf_model=trained["random_forest"].outputs["model"],
        lr_metrics=evaluated["logistic_regression"].outputs["metrics_json"],
        rf_metrics=evaluated["random_forest"].outputs["metrics_json"],
        lr_report=evaluated["logistic_regression"].outputs["report"],
        rf_report=evaluated["random_forest"].outputs["report"],
        lr_plots=evaluated["logistic_regression"].outputs["plots"],
        rf_plots=evaluated["random_forest"].outputs["plots"],
        **creds,
    )
    publish.set_caching_options(False)


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".yaml")
    compiler.Compiler().compile(auto_correction_pipeline, str(out))
    print(f"Compiled {PIPELINE_NAME} -> {out}")
