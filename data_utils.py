"""Data utilities for the HDLM reproduction on the public JHU CSSE COVID-19 time series.

The three global CSV files (confirmed, deaths, recovered) are read, aggregated to
country level, converted to non-negative daily increments and cut into sliding
windows for one-step-ahead forecasting.
"""
import os
import numpy as np
import pandas as pd

JHU_BASE = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
            "csse_covid_19_data/csse_covid_19_time_series/")
SERIES = ("confirmed", "deaths", "recovered")


def download_jhu(data_dir="data"):
    """Download the three JHU CSSE global time-series files if they are missing."""
    import urllib.request
    os.makedirs(data_dir, exist_ok=True)
    for s in SERIES:
        fn = f"time_series_covid19_{s}_global.csv"
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            urllib.request.urlretrieve(JHU_BASE + fn, path)
    return data_dir


def load_country(country="India", data_dir="data", start="2020-01-22", end="2021-08-04"):
    """Return a DataFrame of cumulative and daily counts for one country.

    The end date defaults to 4 August 2021, the last day on which JHU CSSE
    published recovered counts.
    """
    frames = {}
    for s in SERIES:
        df = pd.read_csv(os.path.join(data_dir, f"time_series_covid19_{s}_global.csv"))
        row = df[df["Country/Region"] == country].iloc[:, 4:].sum()
        row.index = pd.to_datetime(row.index, format="%m/%d/%y")
        frames[s] = row.astype(float)
    cum = pd.DataFrame(frames).loc[start:end]
    # cumulative series are forced to be monotone; corrections become zero increments
    cum = cum.cummax()
    daily = cum.diff().fillna(cum.iloc[0]).clip(lower=0)
    daily.columns = [f"new_{c}" for c in daily.columns]
    out = pd.concat([cum, daily], axis=1)
    out["active"] = out["confirmed"] - (out["deaths"] + out["recovered"])
    return out


class LogMinMax:
    """log1p followed by min-max scaling fitted on the training part only."""

    def fit(self, x):
        z = np.log1p(x)
        self.lo, self.hi = z.min(axis=0), z.max(axis=0)
        self.hi = np.where(self.hi - self.lo < 1e-9, self.lo + 1.0, self.hi)
        return self

    def transform(self, x):
        return (np.log1p(x) - self.lo) / (self.hi - self.lo)

    def inverse(self, z, cols=None):
        lo, hi = (self.lo, self.hi) if cols is None else (self.lo[cols], self.hi[cols])
        return np.expm1(z * (hi - lo) + lo)


def make_windows(arr, window, target_cols):
    """Slide a window of length `window` over arr (T x F); target is next-day value."""
    X, y = [], []
    for t in range(window, len(arr)):
        X.append(arr[t - window:t])
        y.append(arr[t, target_cols])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def chronological_split(n, test_frac=0.2, val_frac=0.1):
    """Index ranges for train / validation / test without shuffling."""
    n_test = int(round(n * test_frac))
    n_trval = n - n_test
    n_val = int(round(n_trval * val_frac))
    return (slice(0, n_trval - n_val), slice(n_trval - n_val, n_trval), slice(n_trval, n))
