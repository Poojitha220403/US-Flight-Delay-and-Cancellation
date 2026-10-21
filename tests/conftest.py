"""End-to-end fixtures: run the real pipeline scripts on a small synthetic flight dataset."""
import csv
import datetime as dt
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

AIRLINES = ["Delta Air Lines Inc.", "JetBlue Airways", "Allegiant Air", "Southwest Airlines Co."]
AIRPORTS = ["ATL", "JFK", "DEN", "ORD", "LAX", "CLT"]
N_FLIGHTS = 1500
CANCEL_RATE = 0.05
MONTHS = range(1, 7)  # Jan–Jun 2019


def make_flights(path, seed=42):
    """Kaggle-style CSV. Arrival delay tracks departure delay, as in the real data."""
    rng = random.Random(seed)
    header = ["FL_DATE", "AIRLINE", "ORIGIN", "DEST", "CRS_DEP_TIME", "DEP_DELAY",
              "ARR_DELAY", "CANCELLED", "CANCELLATION_CODE", "DIVERTED", "DISTANCE"]
    stats = {"rows": 0, "cancelled": 0}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(N_FLIGHTS):
            month = MONTHS[i % len(MONTHS)]
            day = rng.randint(1, 28)
            origin, dest = rng.sample(AIRPORTS, 2)
            dep = max(-15.0, rng.gauss(10, 30))
            cancelled = rng.random() < CANCEL_RATE
            arr = "" if cancelled else round(dep + rng.gauss(-3, 8), 1)
            w.writerow([dt.date(2019, month, day).isoformat(), rng.choice(AIRLINES), origin, dest,
                        rng.choice([550, 905, 1230, 1745]), "" if cancelled else round(dep, 1),
                        arr, 1 if cancelled else 0, rng.choice("ABC") if cancelled else "",
                        0, rng.randint(200, 2500)])
            stats["rows"] += 1
            stats["cancelled"] += cancelled
    return stats


def run_script(name, *args, env=None):
    result = subprocess.run(
        [sys.executable, str(SRC / name), *args],
        cwd=ROOT, env={**os.environ, **(env or {})}, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{name} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
    return result


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    base = tmp_path_factory.mktemp("pipeline")
    raw = base / "raw"
    raw.mkdir()
    stats = make_flights(raw / "flights.csv")
    dirs = {name: base / name for name in ["curated", "tables", "models", "stream"]}
    return {"base": base, "raw_glob": str(raw / "*.csv"), "stats": stats, **dirs}


@pytest.fixture(scope="session")
def ingested(workdir):
    run_script("01_ingest_eda.py", env={
        "RAW_DATA_GLOB": workdir["raw_glob"],
        "CURATED_DIR": str(workdir["curated"]),
        "TABLE_DIR": str(workdir["tables"]),
    })
    return workdir


@pytest.fixture(scope="session")
def trained(ingested):
    w = ingested
    run_script("03_predictive_model.py", "--task", "classify", "--algo", "lr",
               "--curated_dir", str(w["curated"]), "--models_dir", str(w["models"]),
               "--predictions_csv", str(w["models"] / "predictions_sample.csv"), "--sample_rows", "50")
    return w


def read_table(workdir, name):
    with open(workdir["tables"] / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))
