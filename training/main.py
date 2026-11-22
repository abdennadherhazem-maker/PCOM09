"""PCOM callback API connected to the five-output LightGBM model.

Based on riehlegroup/pcom-course-setup. Keep the professor's LICENSE in your repo.
Place core.py, forecast.py, train_lightgbm.py and npm_model.json beside this file.
"""
from contextlib import asynccontextmanager
from functools import lru_cache
import io
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import BackgroundTasks, FastAPI
import numpy as np
import pandas as pd
from pydantic import BaseModel
import requests
import uvicorn

from core import BANDS, calendar_end
from train_lightgbm import CSV_FORMAT, CSV_NAMES, expected_counts, load_samples


MODEL_PATH = Path(__file__).resolve().parent / "npm_model.json"

# These are YOUR training target names. Compy's competition description is the
# authority for submission headers. If its names differ, change these strings
# in the same severity order: low, moderate, high, critical, unknown.
OUTPUT_COLUMNS = (
    "target_ghsa_low_count_90d",
    "target_ghsa_moderate_count_90d",
    "target_ghsa_high_count_90d",
    "target_ghsa_critical_count_90d",
    "target_ghsa_unknown_count_90d",
)

# None uses your existing cutoff inference. If the competition input does not
# identify a unique cutoff, set its actual extraction cutoff, e.g. "2026-11-01".
# Do not reuse your training cutoff unless the incoming data uses that cutoff.
CSV_CUTOFF = None

logger = logging.getLogger("uvicorn.error")


@lru_cache(maxsize=1)
def get_model():
    if len(OUTPUT_COLUMNS) != len(BANDS) or len(set(OUTPUT_COLUMNS)) != len(BANDS):
        raise ValueError("Set five distinct OUTPUT_COLUMNS in severity order")
    if any(not isinstance(name, str) or not name.strip() for name in OUTPUT_COLUMNS):
        raise ValueError("OUTPUT_COLUMNS must contain nonempty strings")
    if CSV_CUTOFF is not None:
        calendar_end(CSV_CUTOFF)
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if model.get("data_format") != CSV_FORMAT:
        raise ValueError("This server requires a calendar-cutoff CSV LightGBM model")
    # Validate the saved model before advertising a healthy service.
    expected_counts(model, np.full(len(CSV_NAMES), np.nan))
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="PCOM LightGBM Prediction API", lifespan=lifespan)


class EvaluationBody(BaseModel):
    testdata_download_endpoint: str
    prediction_upload_endpoint: str


def run_model(df: pd.DataFrame) -> pd.DataFrame:
    """Run existing feature extraction and prediction; no training occurs here."""
    model = get_model()
    if df.empty:
        raise ValueError("The input CSV contains no package history")
    # Reuse the same parser and feature extraction as train_lightgbm.py predict.
    # Separate temporary directories allow simultaneous requests safely.
    with TemporaryDirectory(prefix="pcom-input-") as directory:
        input_path = Path(directory) / "input.csv"
        df.to_csv(input_path, index=False)
        kind, samples = load_samples(
            [input_path], [], training=False, csv_cutoff=CSV_CUTOFF
        )
    if kind != model["data_format"] or len(samples) != 1:
        raise ValueError("Each competition request must contain one package window")
    values = expected_counts(model, samples[0].x)[0]
    if len(values) != len(OUTPUT_COLUMNS) or not np.isfinite(values).all():
        raise ValueError("The model must return five finite predictions")
    # The local CLI writes a tuple. Compy receives five separate numeric columns.
    return pd.DataFrame([[float(value) for value in values]], columns=OUTPUT_COLUMNS)


def start_prediction(download_url: str, upload_url: str):
    try:
        with requests.get(download_url, timeout=30) as response:
            response.raise_for_status()
            # Preserve integer strings and blank cells for your feature parser.
            input_data = pd.read_csv(
                io.BytesIO(response.content), sep=",", dtype=str,
                keep_default_na=False, encoding="utf-8-sig",
            )

        prediction_data = run_model(input_data)
        output_csv = prediction_data.to_csv(
            sep=",", decimal=".", header=True, index=False
        )
        files = {"predictions": ("prediction.csv", output_csv, "text/csv")}
        with requests.post(upload_url, files=files, timeout=30) as response:
            response.raise_for_status()
            # A successful callback may return JSON, plain text or an empty body.
            logger.info("Prediction uploaded successfully (HTTP %s)", response.status_code)
    except Exception:
        logger.exception("Prediction job failed")


@app.post("/predict", status_code=200)
async def prediction(payload: EvaluationBody, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        start_prediction,
        payload.testdata_download_endpoint,
        payload.prediction_upload_endpoint,
    )
    return {"status": "prediction started"}


@app.get("/health", status_code=200)
async def health():
    return {"health": "up"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
