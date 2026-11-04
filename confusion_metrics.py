"""Screening metrics from the confusion matrix of the HDLM record-level CNN (Tables 7-8)."""
TP, FN, FP, TN = 5_756_454, 21, 20, 202_028_012


def metrics(tp=TP, fn=FN, fp=FP, tn=TN):
    n = tp + tn + fp + fn
    acc = (tp + tn) / n
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    spec = tn / (tn + fp)
    f1 = 2 * prec * rec / (prec + rec)
    return dict(N=n, accuracy=acc, precision=prec, recall=rec, specificity=spec, f1=f1, error_rate=(fp + fn) / n)


if __name__ == "__main__":
    for k, v in metrics().items():
        print(f"{k:12s} {v:.8f}" if isinstance(v, float) else f"{k:12s} {v:,}")
