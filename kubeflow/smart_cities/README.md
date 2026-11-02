# Smart Cities: Auto-Correction Eligibility Classifier

A small KFP v2 pipeline for the HumAIne Smart Cities pilot (WP6). It supports a demo in which the HumAIne Swarm Assistant runs a Kubeflow training job and then explains the results.

When a citizen's entry on a parking-permit application differs from the correct value, the mistake is either:
- **auto-correctable** (obvious), such as a sound-alike misspelling or a misplaced ZIP code, and can be fixed automatically, or
- **needs operator** (non-obvious), such as a different name or a wrong number, and must go to a human.

The pipeline trains a logistic regression and a random forest to tell the two apart. The data is the synthetic "obvious mistake" dataset built from the 566 anonymised IBM2 applications (`kubeflow/Synthetic mistakes/`: `generate_dataset.py` and the `.docx` description). It has 1,334 training rows and 366 test rows, balanced between the classes.

| Step | What it does |
|------|--------------|
| `load-data` | reads `train.csv` and `test.csv` from MinIO |
| `build-features` | field type, length difference, Levenshtein distance, same characters, same digits |
| `train-model` / `evaluate-model` (logistic regression), `train-model-2` / `evaluate-model-2` (random forest) | accuracy, precision, recall, F1, AUC, confusion matrix, per-field accuracy, HTML report |
| `publish-results` | compares the models, picks the best by F1 and uploads everything to MinIO |

The steps keep their task names in the Kubeflow graph on purpose. The cluster's KFP driver files a task under its display name but looks up upstream outputs by task name, so `set_display_name()` on a step that feeds another step crashes the next step's driver (`nil pointer dereference` in `resolveUpstreamArtifacts`).

The features are deliberately generic. Features that restate how the dataset was generated (the Greek sound-alike map, token reordering) would let both models score 100%. Without them, logistic regression reaches about 0.94 accuracy (it misses some name mistakes) and random forest about 0.997, so there is a real comparison to show.

## MinIO layout

Inputs: `s3://smart-cities-data/auto-correction/{train,test}.csv`

Outputs, written per run to `s3://smart-cities-data/kubeflow/smart-cities-auto-correction-classifier/<run_id>/`:

```
metrics/<model>_metrics.json                 accuracy, precision, recall, F1, AUC, confusion matrix, named outcomes, per-field accuracy
models/<model>.joblib                        {"model", "model_type", "feature_names"}
plots/<model>_confusion_matrix.html          small text tables, read by the assistant
plots/<model>_roc_curve.html
plots/<model>_feature_importance.html
reports/<model>_report.html                  visual report with charts, for people (also shown in the Kubeflow UI)
metadata/run_summary.json                    best model, hyperparameters, data source
metadata/run_parameters.json                 hyperparameters and data paths, read by the assistant
```

The confusion matrix has rows = actual class and columns = predicted class. Because a reader (the assistant included) can easily swap the two, the metrics JSON also has `outcomes` with named counts (for example `needs_operator_wrongly_auto_corrected`, the costly error), and the confusion-matrix page spells out both error counts in a sentence.

`<model>` is `logistic_regression` or `random_forest`, and `<run_id>` is the Kubeflow run ID. This is the layout the assistant's `get_model_metrics`, `get_pipeline_visualization` and `compare_pipeline_runs` tools read. The `plots/` pages stay text-only on purpose, because `get_pipeline_visualization` passes their HTML to the LLM.

## Credentials

The pipeline takes `minio_access_key`, `minio_secret_key` and `minio_session_token` as parameters. When the assistant launches the pipeline, `run_pipeline` fills them with the chatting user's own short-lived MinIO STS keys, so nobody types a password. These temporary keys appear in the run's parameter list and in each step's logs in the Kubeflow UI, and expire with the session, which is fine for a run of a few minutes. To run the pipeline from the Kubeflow UI, paste STS keys yourself.

## Build and publish

```bash
poetry run python kubeflow/smart_cities/auto_correction_pipeline.py      # writes auto_correction_pipeline.yaml
poetry run python kubeflow/smart_cities/upload_demo_assets.py            # CSVs to MinIO + private pipeline in your namespace
```

The upload script asks for your Keycloak username and password (the password is not echoed, printed or stored) and logs in with the `OAUTH_KEYCLOAK_*` settings from `.env`. Pass `--username` to skip the first prompt, or set `KUBEFLOW_TOKEN` to an access token to skip the login altogether. The same token uploads the pipeline and, unless `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` are set, is exchanged for temporary MinIO STS keys to upload the CSVs. The login uses Keycloak's password grant, so the client (default `OAUTH_KEYCLOAK_CLIENT_ID`, change it with `--client-id`) must have "Direct access grants" enabled. The pipeline is uploaded as a private pipeline in your Kubeflow profile namespace (the one you own, looked up from the Kubeflow dashboard), so only members of that profile see it. Use `--namespace <ns>` to choose another profile namespace, or `--shared` to publish to the shared catalog. `KUBEFLOW_HOST` and `MINIO_ENDPOINT` also come from the environment or `.env`. Running the script again adds a new pipeline version. Use `--skip-data` or `--skip-pipeline` to do only one half.

## Demo script

Turn on the Smart Cities and Kubeflow packs, then ask:

1. "Which Kubeflow pipelines can I use?"
2. "Run the Smart Cities auto-correction classifier in experiment `<name>`."
3. "What's the status of that run?" (repeat until it has succeeded, about 3–5 minutes, mostly pip installs)
4. "How did the models do?" or "Compare the two models."
5. "Show me the random forest confusion matrix" or "Which features matter most to the random forest?"
