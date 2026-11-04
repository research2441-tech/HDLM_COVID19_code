# HDLM: Hybrid CNN + stacked LSTM for crowdsourced COVID-19 data

Code accompanying the revised manuscript "A Hybrid Deep Learning Model to Predict the Health Condition of COVID Patients"
(Computer Methods in Biomechanics and Biomedical Engineering).

All files sit in one folder; run every command from inside it.

## Install
```
pip install -r requirements.txt
```

## Reproduce
| Step | Command | Output |
|---|---|---|
| Parameter counts (Table 1, Table 6) | `python check_params.py` | 5,723,457 for the screening CNN; forecaster counts |
| Forecasting experiment (Table 5, Fig. 21) | `python run_experiments.py --out results` | `metrics_runs.csv`, `metrics_summary.csv`, `history_*.csv`, `predictions_hdlm.csv` |
| Figures 15, 16, 21, 22, 23 | `python make_figures.py --results results --out figures` | 300-dpi PNG files |
| Diagrams (Figs. 1, 9, 11) | `python make_diagrams.py` | 300-dpi PNG files |
| Screening metrics (Table 8) | `python confusion_metrics.py` | accuracy, precision, recall, specificity, F1 |

`run_experiments.py` downloads the JHU CSSE global time series on first use
(https://github.com/CSSEGISandData/COVID-19) into `./data`.

## Protocol
* India, 1 April 2020 to 4 August 2021 (last JHU CSSE recovered value).
* Seven-day trailing means of daily cases, deaths and recoveries; 14-day input window; target = next week's mean daily count.
* Chronological split of forecast origins: 72% train, 8% validation (early stopping), 20% test.
* Neural models: Adam (lr 0.001), batch 32, up to 300 epochs, patience 30, seeds 11, 23, 37, 53, 71.
* Baselines: persistence, ARIMA (order by AIC, frozen), Prophet (refitted at each test origin).

## Files
`data_utils.py` data loading and splitting; `models.py` Keras models; `run_experiments.py` training and evaluation;
`make_figures.py` result figures; `make_diagrams.py` method diagrams; `confusion_metrics.py` screening metrics;
`check_params.py` parameter counts; the `*.csv`/`*.json` files are the results reported in the paper.

## Licence
MIT
