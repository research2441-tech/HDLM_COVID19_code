"""Reproduction run of the HDLM forecasting stage on the JHU CSSE India series
(seven-day-ahead forecast of the next week's mean daily counts).

Usage:
    python run_experiments.py --data data --out results --seeds 11 23 37 53 71

Outputs (in --out): metrics_runs.csv, metrics_summary.csv, history_<model>_<seed>.csv,
predictions_hdlm.csv, params.json, environment.json
"""
import argparse, json, os, platform, random, time
import numpy as np
import pandas as pd
import tensorflow as tf

from data_utils import download_jhu, load_country, chronological_split
from models import BUILDERS, build_record_cnn, count_params

FEATS = ["new_confirmed", "new_deaths", "new_recovered"]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = y - p
    mae = np.mean(np.abs(err)); rmse = np.sqrt(np.mean(err ** 2))
    mape = 100 * np.mean(np.abs(err) / np.maximum(y, 1.0))
    r2 = 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)
    acc = 100 * (1 - np.sum(np.abs(err)) / np.sum(np.abs(y)))
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2, ACC=acc)


class EpochTracker(tf.keras.callbacks.Callback):
    """Logs loss and forecast accuracy (100*(1-WAPE), next-week mean daily new confirmed) on train and test each epoch.
    The test set is only observed here; it is never used for early stopping or tuning."""

    def __init__(self, Xtr, ytr, Xte, yte, inv_tr, inv_te):
        super().__init__()
        self.d = (Xtr, ytr, Xte, yte); self.inv = (inv_tr, inv_te); self.rows = []

    def _acc(self, X, y, inv):
        p = self.model.predict(X, verbose=0)
        yt = inv(y)[:, 0]; pt = inv(p)[:, 0]
        return 100 * (1 - np.sum(np.abs(yt - pt)) / np.sum(yt)), float(np.mean((p - y) ** 2))

    def on_epoch_end(self, epoch, logs=None):
        Xtr, ytr, Xte, yte = self.d
        acc_tr, l_tr = self._acc(Xtr, ytr, self.inv[0]); acc_te, l_te = self._acc(Xte, yte, self.inv[1])
        self.rows.append(dict(epoch=epoch + 1, train_loss=l_tr, val_loss=logs.get("val_loss"),
                              test_loss=l_te, train_acc=acc_tr, test_acc=acc_te))


def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s); random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def arima_forecast(series_log, n_fit, origins, H):
    """Order chosen by AIC on the fitting span; parameters then frozen and an H-step forecast is
    issued from every test origin using only data up to that origin."""
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    for p in range(0, 4):
        for d in (0, 1):
            for q in range(0, 3):
                try:
                    r = ARIMA(series_log[:n_fit], order=(p, d, q)).fit()
                    if best is None or r.aic < best[0]:
                        best = (r.aic, (p, d, q), r)
                except Exception:
                    pass
    preds = []
    for o in origins:
        r = best[2].apply(series_log[:o + 1])
        preds.append(np.asarray(r.forecast(H))[-1])
    return np.expm1(np.asarray(preds)), best[1]


def prophet_forecast(dates, series_log, origins, H):
    """Prophet refitted at every test origin on the history available at that origin."""
    from prophet import Prophet
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    preds = []
    for o in origins:
        m = Prophet(weekly_seasonality=False, yearly_seasonality=False, daily_seasonality=False)
        m.fit(pd.DataFrame({"ds": dates[:o + 1], "y": series_log[:o + 1]}))
        fc = m.predict(pd.DataFrame({"ds": [dates[o + H]]}))
        preds.append(np.expm1(fc["yhat"].values[0]))
    return np.asarray(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--country", default="India"); ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--start", default="2020-04-01", help="first day kept (drops the zero-inflated pre-epidemic weeks)")
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    ap.add_argument("--skip-baselines", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    download_jhu(a.data)
    df = load_country(a.country, a.data, start=a.start)
    raw = df[FEATS].values
    W, H, K = a.window, a.horizon, 7
    # 7-day trailing mean removes the weekly reporting cycle; S[o+H] is next week's mean daily count
    S = pd.DataFrame(raw).rolling(K).mean().values
    L = np.log1p(S)
    origins = np.arange(W + K - 2, len(raw) - H)
    X = np.stack([L[o - W + 1:o + 1] - L[o] for o in origins]).astype("float32")
    y = np.stack([L[o + H] - L[o] for o in origins]).astype("float32")
    tr, va, te = chronological_split(len(origins))
    sd = y[tr].std(axis=0) + 1e-6
    X = X / sd; y = y / sd
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    base = L[origins]

    def make_inv(b):
        return lambda z: np.expm1(b + np.clip(z * sd, -8, 8))
    inv_tr, inv_te = make_inv(base[tr]), make_inv(base[te])
    o_te = origins[te]
    test_dates = df.index[o_te + H]
    yte_raw = S[o_te + H]
    cum_o = df["confirmed"].values[o_te]
    info = dict(n_days=len(df), start=str(df.index[0].date()), end=str(df.index[-1].date()),
                window=W, horizon=H, smoothing=K, n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
                train_origins=f"{df.index[origins[tr][0]].date()}..{df.index[origins[tr][-1]].date()}",
                val_origins=f"{df.index[origins[va][0]].date()}..{df.index[origins[va][-1]].date()}",
                test_origins=f"{df.index[o_te[0]].date()}..{df.index[o_te[-1]].date()}",
                test_targets=f"{test_dates[0].date()}..{test_dates[-1].date()}")
    rows, params, preds_keep = [], {}, {}
    for name, builder in BUILDERS.items():
        for s in a.seeds:
            set_seed(s)
            m = builder(W, len(FEATS), len(FEATS))
            params[name] = count_params(m)
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            tracker = EpochTracker(Xtr, ytr, Xte, yte, inv_tr, inv_te)
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience,
                                                  restore_best_weights=True)
            t0 = time.time()
            h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=32,
                      shuffle=True, verbose=0, callbacks=[tracker, es])
            fit_s = time.time() - t0
            pd.DataFrame(tracker.rows).to_csv(
                os.path.join(a.out, f"history_{name.split(' ')[0]}_{s}.csv"), index=False)
            t1 = time.time(); p = m.predict(Xte, verbose=0); inf_ms = 1000 * (time.time() - t1) / len(Xte)
            p_raw = inv_te(p)
            r = dict(model=name, seed=s, epochs=len(h.history["loss"]),
                     best_epoch=int(np.argmin(h.history["val_loss"]) + 1), fit_s=fit_s, infer_ms=inf_ms)
            for j, f in enumerate(FEATS):
                for k, v in metrics(yte_raw[:, j], p_raw[:, j]).items():
                    r[f"{f}_{k}"] = v
            r["cum_confirmed_MAPE"] = metrics(cum_o + H * yte_raw[:, 0], cum_o + H * p_raw[:, 0])["MAPE"]
            rows.append(r)
            if name.startswith("HDLM"):
                preds_keep[s] = p_raw
            print(name, s, round(r["new_confirmed_ACC"], 2), round(r["new_confirmed_R2"], 4), flush=True)
    # statistical / naive baselines (deterministic)
    if not a.skip_baselines:
        persist = S[o_te][:, 0]
        logc = L[:, 0]; n_fit = origins[va][-1] + 1
        ar_pred, order = arima_forecast(np.nan_to_num(logc), n_fit, o_te, H)
        info["arima_order"] = order
        valid = ~np.isnan(logc)
        first = int(np.argmax(valid))
        pr_pred = prophet_forecast(df.index[first:], logc[first:], o_te - first, H)
        for name, pc in [("Persistence (naive)", persist), (f"ARIMA{order}", ar_pred), ("Prophet", pr_pred)]:
            r = dict(model=name, seed=0)
            for k, v in metrics(yte_raw[:, 0], pc).items():
                r[f"new_confirmed_{k}"] = v
            r["cum_confirmed_MAPE"] = metrics(cum_o + H * yte_raw[:, 0], cum_o + H * pc)["MAPE"]
            rows.append(r)
            print(name, round(r["new_confirmed_ACC"], 2), round(r["new_confirmed_R2"], 4), flush=True)
    runs = pd.DataFrame(rows); runs.to_csv(os.path.join(a.out, "metrics_runs.csv"), index=False)
    summ = runs.groupby("model", sort=False).agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]; summ.to_csv(os.path.join(a.out, "metrics_summary.csv"))
    # keep HDLM predictions (mean over seeds) for the forecast figure
    pm = np.mean(np.stack(list(preds_keep.values())), axis=0)
    pd.DataFrame({"date": test_dates, **{f"actual_{f}": yte_raw[:, j] for j, f in enumerate(FEATS)},
                  **{f"pred_{f}": pm[:, j] for j, f in enumerate(FEATS)},
                  "cum_confirmed_at_origin": cum_o}).to_csv(os.path.join(a.out, "predictions_hdlm.csv"), index=False)
    cnn = build_record_cnn()
    params["Record CNN (Table 1)"] = count_params(cnn)
    with open(os.path.join(a.out, "params.json"), "w") as f:
        json.dump({k: dict(trainable=v[0], non_trainable=v[1]) for k, v in params.items()}, f, indent=2)
    env = dict(python=platform.python_version(), tensorflow=tf.__version__, numpy=np.__version__,
               pandas=pd.__version__, platform=platform.platform(), processor=platform.processor(),
               cpu_count=os.cpu_count(), **info)
    with open(os.path.join(a.out, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
