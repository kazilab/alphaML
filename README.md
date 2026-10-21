The alphaML is developed for research purposes only.
The platform has been tested using pharmacogenomic data from cell lines and human patients.
The platform comes with no guarantee and must be used with the user's own responsibility.

# alphaML 2.0

alphaML 2.0 is an AutoML web application for CLETE (Clear, Legible, Explainable, Transparent and Elucidative)
binary classification models from tabular data such as gene expression. Upload your data, and alphaML selects
features, finds the best algorithm, tunes it, explains the model and predicts new samples, all from the browser.

The [user handbook](alphaml/web/static/handbook.html) is available at `/static/handbook.html`, linked from the
application header and footer even before login. It covers input preparation, example files, feature selection,
training, prediction, results, scientific limitations, privacy and troubleshooting, with a Print / save PDF option.
Regenerate its synthetic CSV examples with `python -m alphaml.web.build_handbook_examples`.

## Start with Docker (recommended)

With [Docker Desktop](https://www.docker.com/products/docker-desktop/) running:

- Windows: double-click `start-alphaml.cmd` (or run it in a terminal)
- macOS / Linux: `./start-alphaml.sh`

The script builds the image (the first time takes a few minutes), starts the container and opens
http://localhost:8765 in your browser. If another program already uses port 8765, it picks the next free port and
tells you which; run it again any time to reopen alphaML. For a specific port: `start-alphaml.cmd 8800`.

Without the script:

```
docker compose up -d --build      # start, then open http://localhost:8765
docker compose down               # stop
```

The app is published on `localhost` only, so other computers cannot reach it. With `docker compose`, choose the
port with `ALPHAML_PORT=8800 docker compose up -d`.

## Start without Docker

Python 3.10 or newer:

```
pip install .
alphaml-web                       # then open http://127.0.0.1:8000
alphaml-web --home D:\my_alphaml  # keep data sets and results in another folder
```

## Where data is stored

All uploaded data sets and all job results are kept in one folder:

| How alphaML runs | Folder |
|---|---|
| Docker (`start-alphaml.cmd`, `start-alphaml.sh` or `docker compose`) | `tmp` next to these files, e.g. `E:\alphaML\tmp`. Set `ALPHAML_DATA_DIR` before starting to use another folder. |
| Without Docker (`alphaml-web`) | `Documents\alphaML` in your user folder, or the folder given with `--home` / `ALPHAML_HOME` |
| Python API | input files are read from where you point `train()`/`predict()`; results go to the `result_path` you give |

Inside that folder:

```
datasets/<id>/      uploaded input files (labeled_data.csv, data_labels.csv, ...) and dataset.json
jobs/<id>/results/  everything a job produces: model bundle, scores, predictions, plots, leaderboard
jobs/<id>/inputs/   test data (and model) uploaded for a prediction job
jobs/<id>/run.log   the job log; job.json holds its settings and status
```

Deleting a data set or job in the web app deletes its folder. To start from scratch, stop alphaML and delete the
folder. Back it up like any other folder. In the Docker image the folder is mounted at `/data`. With accounts on,
`users.db` in the same folder holds the accounts (passwords only as salted scrypt hashes).

## Accounts and data retention

With accounts on (the default in Docker; `alphaml-web --auth` or `ALPHAML_AUTH=1` otherwise):

- Everyone creates an account (email and password) and logs in. Each user sees only their own data sets, jobs
  and results; other users' items cannot be listed, opened or downloaded.
- Users can start a job, log out or close the browser, and log in later to see the results.
- Data sets and jobs are **deleted automatically 14 days** after they were uploaded or submitted (the clean-up
  runs every hour; a job still running then is deleted after it finishes, and a data set while no job uses it).
  Each item shows its deletion date. Change the period with
  `ALPHAML_RETENTION_DAYS` (`0` keeps everything).
- The example data set is shared by all users (read-only, not deleted).
- Each user can have at most 3 queued or running jobs (`ALPHAML_MAX_JOBS_PER_USER`).
- Uploading model files for prediction is off, because loading one can run code on the server; users predict
  with the models they trained.
- Users can change their password and delete their account with all their data from the menu at the top right.
- **Privacy (GDPR):** creating an account requires ticking a box: the user agrees that alphaML stores their email
  address and deletes data sets and results after the retention period, will upload only anonymised data (no
  personal or sensitive information such as data identifying patients), and is responsible for the data they
  upload. The agreement links to the privacy notice (`#/privacy`) and to the
  [GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj). When and to which version of the terms each user agreed is
  stored with the account. Accounts created by the administrator, and all users after the terms change (including
  a new retention period), must agree before they can use data again; deleting an account never requires it. Set
  `ALPHAML_OPERATOR` and `ALPHAML_PRIVACY_CONTACT` so the notice says who runs the server and how to reach them.
  Have the texts checked against your institution's data-protection requirements.
- A forgotten password is reset with *Forgot your password?* on the login screen: alphaML emails a link that
  works once, for 60 minutes (at most 3 emails per address and hour; the reply does not reveal whether an
  address has an account). This needs the SMTP settings in `.env` (see `.env.example`); without them, the
  administrator resets passwords on the command line.

Without accounts (local `alphaml-web` by default) everything is visible to whoever opens the app and nothing is
deleted automatically.

To limit who can sign up, set `ALPHAML_SIGNUP_CODE` (people need the code to create an account) or
`ALPHAML_ALLOW_SIGNUP=0` (only the administrator creates accounts). Administrators manage accounts on the
command line (with Docker, prefix the commands with `docker exec -it alphaml`):

```
alphaml-web users list                      # accounts and how many data sets and jobs each has
alphaml-web users add EMAIL                 # create an account (asks for the password)
alphaml-web users reset-password EMAIL      # set a new password, e.g. when a user forgot theirs
alphaml-web users delete EMAIL              # delete the account and all its data
alphaml-web users claim EMAIL               # give data from before accounts were on to this account
```

To let other computers use the server, serve it over HTTPS so that passwords are encrypted on the network. With
a Cloudflare Tunnel (a public host name routed to `http://localhost:8765`), the port stays on `127.0.0.1`; set
in `.env`:

```
ALPHAML_ALLOWED_HOSTS=localhost,127.0.0.1,alphaml.example.org
ALPHAML_PUBLIC_URL=https://alphaml.example.org
ALPHAML_COOKIE_SECURE=1
ALPHAML_CLIENT_IP_HEADER=CF-Connecting-IP
```

Then restart with `start-alphaml.cmd` (or `docker compose up -d`). Without these, browsers that use the public
name get "Invalid host header".

Files are uploaded in 8 MB chunks, because a proxy such as Cloudflare refuses requests over 100 MB (on its Free
and Pro plans); files of any size up to `ALPHAML_MAX_UPLOAD_MB` therefore work through the public address too.
Interrupted chunks are resent automatically, and uploads that are started but never used are deleted after a day.

## Using alphaML

**Train.** Upload a data set (or click *Load example data*), pick the class column and the two classes, and set
up feature selection: up to four methods applied in sequence. Then choose:

- **AutoML** (default): alphaML compares the algorithms and selects the model, as described below.
- **Single algorithm**: train one algorithm, with hyperparameters from an Optuna, Bayesian or grid search, or
  predefined values.

Optionally add feature importance, SHAP and LIME explanations and a ROC curve from repeated cross-validation.
Jobs run in the background; the *Jobs* page shows live progress and the log, then scores, plots, the AutoML
leaderboard and every result file, and downloads everything as a zip.

**Predict.** Choose a finished training job (or upload a model bundle) and upload test data. Features are
selected and normalized exactly as during training.

### How AutoML selects the model

1. **Screen.** Every selected algorithm (AdaBoost, CatBoost, ElasticNet, ExtraTrees, GradientBoosting,
   HistGradientBoosting, LASSO, LightGBM, MLPC (neural network), RandomForest, Ridge, SVC, NuSVC, XGBoost) is
   cross-validated with
   its predefined parameters.
2. **Tune.** The best *Algorithms to tune* (default 3) each get an Optuna hyperparameter search, sharing the
   *time budget* (default 10 minutes; each search also stops after *Optuna trials per algorithm*).
3. **Combine.** Optionally, a soft-voting ensemble averages the probabilities of the tuned algorithms.

All candidates are scored on the same stratified cross-validation folds of the training data, by the metric you
choose: Cohen's kappa × MCC (default), ROC AUC, balanced accuracy or F1. With *Control fitting*, candidates
whose training score is much higher than their validation score rank lower. The best candidate is trained on the
training data and only then evaluated on the held-out test set, so the test scores are not biased by the
selection. The leaderboard (`*_AutoML_leaderboard.csv` and `.png`) lists every candidate.

### Input files

CSV (or tab-separated `.tsv`/`.txt`) files with sample names in the first column:

| File | Required | Content |
|---|---|---|
| Labeled data | yes | samples × features (e.g. genes) |
| Labels | yes | samples × label columns (e.g. one column per drug with `resistant`/`sensitive`) |
| Data for feature selection | no | samples × features used by unsupervised selection methods (default: the labeled data) |
| Unlabeled data | no | samples × features, predicted with the final model after training |
| Suggested features | no | one column of feature names, for the *Suggested* selection method |

Labels are matched to data by sample name (or by row order if the names differ). Samples with any other label
are left out.

### Results

Each training job writes the model bundle (`*_bundle.pkl`: the model with its selected features, normalization
and class names, used for prediction), the selected features, test-set scores (`*_scores.csv`, including the
NegLog2-RMSL score), test predictions, a confusion matrix, a training curve (for single algorithms), the AutoML
leaderboard, and the explanations and ROC curve you asked for.

## Settings

| `alphaml-web` option | Environment variable | Default | Meaning |
|---|---|---|---|
| `--host` | `ALPHAML_HOST` | `127.0.0.1` | Interface to listen on. `127.0.0.1` means this computer only. |
| `--port` | `ALPHAML_PORT` | `8000` (`8765` with Docker) | Port. |
| `--home` | `ALPHAML_HOME` | `Documents\alphaML` (Docker: see [Where data is stored](#where-data-is-stored)) | Folder for data sets and job results. |
| `--max-jobs` | `ALPHAML_MAX_JOBS` | `1` | Jobs that may run at the same time. Each job already uses many CPU cores. |
| | `ALPHAML_MAX_UPLOAD_MB` | `2048` | Largest accepted file. |
| | `ALPHAML_UPLOAD_CHUNK_MB` | `8` | Size of the pieces files are uploaded in (at most 64; must stay under the proxy's request limit). |
| | `ALPHAML_THREADS` | CPU cores, at most 16 | Threads per model fit for the multi-threaded algorithms. |
| `--auth` | `ALPHAML_AUTH` | off (`1` in Docker) | Require accounts; each user sees only their own data. |
| `--retention-days` | `ALPHAML_RETENTION_DAYS` | `14` with accounts, else `0` | Delete data sets and results this many days after creation; `0` keeps them. |
| | `ALPHAML_SIGNUP_CODE` | none | Code needed to create an account. |
| | `ALPHAML_ALLOW_SIGNUP` | `1` | `0`: only the administrator creates accounts. |
| | `ALPHAML_MAX_JOBS_PER_USER` | `3` | Queued and running jobs per user, with accounts. |
| | `ALPHAML_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Host names the server answers to; `*` for any. |
| | `ALPHAML_COOKIE_SECURE` | `0` | `1` when the server is reached over HTTPS. |
| | `ALPHAML_SMTP_HOST`, `_PORT`, `_SECURITY`, `_USERNAME`, `_PASSWORD`, `_FROM` | none, `587`, `starttls` | Outgoing email for password-reset links. In Docker, set them in `.env` (git-ignored). |
| | `ALPHAML_PUBLIC_URL` | address of the request | Address used in emailed links; set it when the server has a domain name. |
| | `ALPHAML_CLIENT_IP_HEADER` | none | Header with the visitor's address behind a proxy or tunnel (`CF-Connecting-IP` for Cloudflare), so login limits apply per visitor. Only while the port is reachable through that proxy alone. |
| | `ALPHAML_OPERATOR`, `ALPHAML_PRIVACY_CONTACT` | none | Who runs the server and how to contact them about personal data, shown in the privacy notice. |
| | `ALPHAML_ALLOW_MODEL_UPLOAD` | `1`, or `0` with accounts | Whether users may upload model bundles for prediction (see Security). |

With Docker, set the environment variables in `docker-compose.yml` or before `docker compose up`. Parallel
cross-validation adapts to the container's CPU limit.

## Python API

```python
from alphaml import train, predict

summary = train('labeled_data.csv', 'data_labels.csv', result_path='results/',
                col_label='Trametinib', pos_class='resistant', neg_class='sensitive',
                features='HVF', model_name='AutoML', automl_time_budget=10, run_roc=True)
print(summary['selected_model'], summary['leaderboard'][:3])

predict('test_data.csv', result_path='predictions/', model='results/<prefix>_bundle.pkl')
```

`model_name` can also be one algorithm, e.g. `'XGBoost'` with `param_search='Optuna'`.

## Security

- Without accounts, anyone who can open the app sees all data: keep such a server on `localhost` (the default).
  For several users, turn accounts on (see [Accounts and data retention](#accounts-and-data-retention)).
- Model bundles are Python pickles, and loading a pickle can run code. Only upload bundles you created or trust.
  Uploads are off when accounts are on.
- The server only accepts changes (uploads, jobs) from its own pages, and only answers to the host names in
  `ALPHAML_ALLOWED_HOSTS`, so other websites open in your browser cannot use a local alphaML server.
- Passwords are stored as salted scrypt hashes and login sessions as hashed tokens; repeated failed logins are
  blocked for 15 minutes. Serve alphaML over HTTPS when other computers connect to it.

## What is new in 2.0

- **AutoML** chooses the algorithm, its hyperparameters and, optionally, an ensemble.
- **Web application** in the browser replaces the desktop windows, with background jobs that can be queued and
  cancelled, and a **Docker** image that runs it on localhost.
- **Removed:** TabNet (PyTorch), the DeepLink/ModelX knockoff feature selection (TensorFlow), the desktop GUIs,
  the `aml()`/`pred()` functions and their `~/Documents` folders, and the Test_Briefly mode (replaced by the
  AutoML leaderboard). Models from alphaML 0.x cannot be used for prediction; retrain them in 2.0.
- **Faster:** training curves from a single fit instead of one refit per iteration count; cross-validation
  folds in parallel; SVC/NuSVC searches without the probability calibration that does not change predictions;
  vectorized feature selection (random projection no longer builds a features × features matrix).
- **Fixes** (results can differ from 0.x): the MCC formula; ROC curves from probabilities instead of predicted
  labels; prediction-time standardization now per sample, as in training; HVF for log10/ln data; CatBoost no
  longer picks its best iteration on the test set; worse-than-chance models no longer score well in the
  kappa/MCC search objective; seeded sampling, random projection and Optuna for reproducible runs.
