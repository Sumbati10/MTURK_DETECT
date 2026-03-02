# MTURK_DETECT (MISSING UNTUKED JOBS SHOWING WORK DONE)

This folder contains a small set of Python scripts extracted from `Missing_ltds-Copy1.ipynb`.

The goal is to:

- Find `SummaryAPICache` rows for a given `day` that look **incomplete** (missing land activity and/or missing polygon), and that are not already locked / created.
- For each affected tractor/asset, fetch `LiveTrackingData` points for that day.
- Run a DBSCAN-based heuristic to detect whether **work was likely done**.
- Output a CSV/Parquet of the flagged rows.

## Files

All scripts live in `services/`:

- `services/django_setup.py`
  - Bootstraps Django so these scripts can import your Django apps.
  - Supports:
    - `--django-settings` (sets `DJANGO_SETTINGS_MODULE`)
    - `--django-path` (prepends your Django project root to `sys.path`)

- `services/step1_problem_qs.py`
  - Notebook Step 1
  - Builds the Django queryset that identifies “problem” `SummaryAPICache` rows for the requested day.

- `services/step2_work_detection.py`
  - Notebook Step 2
  - Contains the DBSCAN + geometry/time heuristics:
    - `identify_work_done_for_asset_day_inline(df, ...)`

- `services/main.py`
  - Notebook Step 3 (entrypoint)
  - Orchestrates:
    - Django setup
    - `problem_qs`
    - fetching `Asset` and `LiveTrackingData`
    - running the work detector
    - writing output

## Requirements

### 1) You must have access to the Django project that contains `ht`

These scripts import:

- `ht.apps.reports.models`
- `ht.apps.real_time_data.models`
- `ht.apps.timescaledb.models`

So you need:

- The Django repo checked out somewhere on disk
- A Python environment where that repo’s dependencies are installed

### 2) Python packages

At runtime the scripts need:

- `pandas`
- `numpy`
- `scikit-learn`
- `django`

Optional:

- `scipy` (used for convex hull area; if missing, hull area becomes `0.0` and clusters may fail `min_cluster_area_ha`)

## How to run

### Run via `python -m` (recommended)

Run from the **parent directory** of `services/`:

```bash
python3 -m services.main \
  --day 2026-02-03 \
  --output flagged.csv \
  --django-settings your_project.settings \
  --django-path /abs/path/to/your/django/repo
```

- `--day` must be `YYYY-MM-DD`.
- `--output`:
  - If it ends with `.csv`, output is CSV.
  - Otherwise output is Parquet.

### Alternative: environment variables (instead of flags)

If you already export `DJANGO_SETTINGS_MODULE` and/or have your repo on `PYTHONPATH`, you can omit the flags.

Example:

```bash
export DJANGO_SETTINGS_MODULE=your_project.settings
export PYTHONPATH=/abs/path/to/your/django/repo
python3 -m services.main --day 2026-02-03 --output flagged.csv
```

## Output

The output file contains one row per “flagged” `SummaryAPICache` where the work detector returned `worked=True`.

Important columns:

- `sac_id`
- `tractor_id`
- `asset_uuid`
- `reason_codes`
- `points_used`
- `path_km`
- `moving_ratio`
- `top_cluster_*` metrics

## Troubleshooting

- `ModuleNotFoundError: No module named 'ht'`
  - Provide `--django-path /abs/path/to/django/repo` (the repo that contains the `ht/` package).

- `django.core.exceptions.ImproperlyConfigured`
  - Provide `--django-settings your_project.settings` or export `DJANGO_SETTINGS_MODULE`.

- SciPy missing
  - If you see very few/no clusters flagged, install SciPy in your environment so convex hull area can be computed.
