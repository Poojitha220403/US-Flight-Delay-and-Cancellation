import csv
import json

import pytest

from conftest import AIRLINES, MONTHS, read_table, run_script

TABLES = ["by_airline", "by_month", "routes_most_delayed", "cancellations_by_code",
          "origin_top_delay", "dest_top_delay", "arr_delay_summary",
          "distance_delay_correlation", "dep_vs_arr_by_airline", "by_day_of_week"]


def flown(stats):
    return stats["rows"] - stats["cancelled"]


# ---------- 01_ingest_eda.py ----------

def test_ingest_writes_all_ten_tables(ingested):
    for name in TABLES:
        assert (ingested["tables"] / f"{name}.csv").exists(), name


def test_ingest_writes_partitioned_parquet(ingested):
    partitions = sorted(p.name for p in (ingested["curated"] / "year=2019").iterdir() if p.is_dir())
    assert partitions == [f"month={m}" for m in MONTHS]


def test_airline_table_counts_every_flown_flight(ingested):
    rows = read_table(ingested, "by_airline")
    assert {r["airline"] for r in rows} == set(AIRLINES)
    assert sum(int(r["n"]) for r in rows) == flown(ingested["stats"])
    delays = [float(r["avg_arr_delay"]) for r in rows]
    assert delays == sorted(delays, reverse=True)  # most delayed first


def test_monthly_table_and_summary_agree(ingested):
    months = read_table(ingested, "by_month")
    assert [(int(r["year"]), int(r["month"])) for r in months] == [(2019, m) for m in MONTHS]
    summary = {r["summary"]: r["arr_delay"] for r in read_table(ingested, "arr_delay_summary")}
    assert int(float(summary["count"])) == sum(int(r["n"]) for r in months)


def test_distance_correlation_is_a_valid_coefficient(ingested):
    (row,) = read_table(ingested, "distance_delay_correlation")
    assert -1.0 <= float(row["distance_arr_delay_corr"]) <= 1.0


@pytest.mark.xfail(strict=True, reason="Known limitation (README): cleaning drops rows with no arr_delay, "
                                       "which removes every cancelled flight before the cancellation table")
def test_cancellation_table_counts_cancelled_flights(ingested):
    rows = read_table(ingested, "cancellations_by_code")
    counted = sum(int(r["count"]) for r in rows if r["cancellation_code"])
    assert counted == ingested["stats"]["cancelled"]


# ---------- 03_predictive_model.py ----------

def test_classifier_reports_strong_roc_auc(trained):
    line = (trained["models"] / "metrics.txt").read_text().strip().splitlines()[-1]
    assert line.startswith("TASK=classify, ALGO=lr, THRESH=15.0")
    auc = float(line.split("ROC-AUC=")[1])
    assert 0.8 < auc <= 1.0  # arrival delay tracks departure delay in the fixture


def test_model_artifacts_are_consistent(trained):
    models = trained["models"]
    assert (models / "flight_delay_classify_lr" / "metadata").is_dir()
    card = json.loads((models / "model_card.json").read_text())
    assert card["task"] == "classify" and card["label_col"] == "delayed"
    assert card["n_train"] + card["n_test"] == flown(trained["stats"])
    with open(models / "confusion_matrix.csv", newline="") as f:
        assert sum(int(r["count"]) for r in csv.DictReader(f)) == card["n_test"]
    with open(models / "predictions_sample.csv", newline="") as f:
        sample = list(csv.DictReader(f))
    assert len(sample) == 50 and set(sample[0]) == {"delayed", "prediction", "probability"}


# ---------- 04a_make_stream_batches.py ----------

def test_stream_batches_cover_the_last_three_months(ingested):
    w = ingested
    run_script("04a_make_stream_batches.py", "--curated_dir", str(w["curated"]),
               "--stream_dir", str(w["stream"]), "--months", "3", "--rows_per_file", "200")
    batches = sorted(w["stream"].glob("batch_*.csv"))
    assert batches and batches[0].name == "batch_0001.csv"
    rows = []
    for b in batches:
        with open(b, newline="") as f:
            batch = list(csv.DictReader(f))
        assert 0 < len(batch) <= 200
        rows += batch
    assert {"flight_date", "airline", "origin", "dest", "arr_delay", "dep_delay"} <= set(rows[0])
    assert {int(r["month"]) for r in rows} == {4, 5, 6}
    expected = sum(int(r["n"]) for r in read_table(w, "by_month") if int(r["month"]) >= 4)
    assert len(rows) == expected
    dates = [r["flight_date"] for r in rows]
    assert dates == sorted(dates)  # batches are in event-time order
