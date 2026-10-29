"""
P-396 Machine Learning-Based Stock Market Trend Prediction System (RELIANCE)
Universal Pipeline: Runs in VS Code (Terminal / Jupyter) and Google Colab.
Primary Data Source: Yahoo Finance (yfinance).
"""

import os
import sys
import subprocess
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. Configuration & Reproducibility
# ---------------------------------------------------------
TICKER      = 'RELIANCE.NS'
START_DATE  = '2015-01-01'
END_DATE    = '2026-10-19'      # Frozen for exact reproducibility
SEED        = 42
SEQ_LEN     = 20                # LSTM lookback window (days)
WARMUP      = 30                # Warmup days skipped for technical indicator stability
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
OUT_DIR     = 'outputs'

os.makedirs(OUT_DIR, exist_ok=True)

# Set seeds for strict determinism
random.seed(SEED)
np.random.seed(SEED)

# Detect Deep Learning Backend: TensorFlow (Colab) or PyTorch (local VS Code)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    DL_BACKEND = 'tensorflow'
except ImportError:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    DL_BACKEND = 'pytorch'

print(f"=== P-396 Stock Trend Pipeline ===")
print(f"Ticker: {TICKER} | Start: {START_DATE} | End: {END_DATE} | Seed: {SEED}")
print(f"Deep Learning Engine: {DL_BACKEND.upper()} (Universal Compatibility)")

# ---------------------------------------------------------
# 2. Data Ingestion: Yahoo Finance with Local Snapshot Fallback
# ---------------------------------------------------------
def load_market_data(ticker=TICKER, start=START_DATE, end=END_DATE, cache_file='data_snapshot_yahoo.csv'):
    df = None
    end_fetch = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if end else None

    # 1. Attempt Yahoo Finance download
    try:
        import yfinance as yf
        print(f"Fetching data from Yahoo Finance for {ticker}...")
        raw_df = yf.download(ticker, start=start, end=end_fetch, auto_adjust=False, progress=False)
        if raw_df is not None and not raw_df.empty:
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = [c[0] for c in raw_df.columns]
            raw_df.to_csv(cache_file)
            df = raw_df
            print(f"Successfully downloaded {len(df)} rows from Yahoo Finance.")
    except Exception as e:
        print(f"Notice: Yahoo Finance online fetch encountered: {e}")

    # 2. Fallback to cached local snapshot
    if df is None or df.empty:
        if os.path.exists(cache_file):
            print(f"Loading from local cache: {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        elif os.path.exists('RELIANCE.parquet') or os.path.exists('_dl/RELIANCE.parquet'):
            p = 'RELIANCE.parquet' if os.path.exists('RELIANCE.parquet') else '_dl/RELIANCE.parquet'
            print(f"Loading from parquet fallback: {p}")
            df = pd.read_parquet(p)
        else:
            raise RuntimeError("Could not load data from Yahoo Finance or local cache.")

    # Standardize column names
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

    req = ['date', 'open', 'high', 'low', 'close', 'volume']
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'. Found: {list(df.columns)}")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    df['date'] = df['date'].dt.normalize()

    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Data Hygiene
    n0 = len(df)
    df = df.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values('date')
    df = df.drop_duplicates(subset='date', keep='last')
    df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1) & (df['high'] >= df['low'])]
    df = df[df['volume'] > 0]
    print(f"Clean records remaining: {len(df)} (filtered {n0 - len(df)} bad/missing rows)")

    raw = df.set_index('date')[['open', 'high', 'low', 'close', 'volume']].rename(columns=str.title)
    raw.index.name = 'Date'
    raw = raw[raw.index >= pd.Timestamp(START_DATE)]
    if END_DATE:
        raw = raw[raw.index <= pd.Timestamp(END_DATE)]

    assert raw.index.is_monotonic_increasing and raw.index.is_unique and raw.notna().all().all()
    print(f"Date range: {raw.index.min().date()} to {raw.index.max().date()} ({len(raw)} trading days)")
    return raw

raw_prices = load_market_data()

# ---------------------------------------------------------
# 3. Technical Indicator Engineering & Target Formulation
# ---------------------------------------------------------
RAW_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
IND_COLS = ['SMA_10', 'EMA_10', 'RSI_14', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower']

def add_indicators(prices):
    out = prices[RAW_COLS].copy()
    c = out['Close']
    out['SMA_10'] = c.rolling(10).mean()
    out['EMA_10'] = c.ewm(span=10, adjust=False).mean()

    # Wilder's Smoothing for RSI
    delta = c.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    out['RSI_14'] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-9))))

    # MACD (12, 26, 9)
    out['MACD'] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out['MACD_signal'] = out['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    mid = c.rolling(20).mean()
    sd  = c.rolling(20).std()
    out['BB_upper'] = mid + 2 * sd
    out['BB_lower'] = mid - 2 * sd
    return out.replace([np.inf, -np.inf], np.nan)

ind_df = add_indicators(raw_prices)

# Next-day Target: 1 if tomorrow Close > today Close, else 0
tomorrow = raw_prices['Close'].shift(-1)
target_series = (tomorrow > raw_prices['Close']).astype(float).where(tomorrow.notna())

# Skip warmup period to allow rolling indicators to stabilize
data = ind_df.assign(target=target_series).iloc[WARMUP:].dropna()
X = data[RAW_COLS + IND_COLS]
y = data['target'].astype(int)

FEATURE_SETS = {
    'raw': RAW_COLS,
    'indicator': RAW_COLS + IND_COLS
}

print(f"Total labelled days: {len(data)} | Positive class (UP) ratio: {y.mean():.3f}")

# ---------------------------------------------------------
# 4. Strict Chronological Split (70 / 15 / 15)
# ---------------------------------------------------------
n = len(data)
i1 = int(n * TRAIN_FRAC)
i2 = int(n * (TRAIN_FRAC + VAL_FRAC))

Xtr, Xva, Xte = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
ytr, yva, yte = y.iloc[:i1], y.iloc[i1:i2], y.iloc[i2:]

train_majority = int((ytr == 1).mean() >= 0.5)
baseline = float((yte == train_majority).mean())

split_info = (
    f"Ticker: {TICKER}\n"
    f"Train: {len(Xtr)} rows ({Xtr.index.min().date()} to {Xtr.index.max().date()})\n"
    f"Val:   {len(Xva)} rows ({Xva.index.min().date()} to {Xva.index.max().date()})\n"
    f"Test:  {len(Xte)} rows ({Xte.index.min().date()} to {Xte.index.max().date()})\n"
    f"Majority Baseline (Train majority={train_majority}): {baseline:.3f}\n"
)
print("\n" + split_info)

for p in [OUT_DIR, '.']:
    with open(os.path.join(p, 'chronological_split_dates.txt'), 'w') as f:
        f.write(split_info)

# ---------------------------------------------------------
# 5. Model Definitions & Validation Tuning
# ---------------------------------------------------------
def make_model(name, params, pos_weight):
    if name == 'DecisionTree':
        return DecisionTreeClassifier(class_weight='balanced', random_state=SEED, **params)
    elif name == 'RandomForest':
        return RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1, **params)
    elif name == 'SVM':
        return SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=SEED, **params)
    elif name == 'XGBoost':
        return XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=SEED, n_jobs=-1, **params)

GRID = {
    'DecisionTree': [dict(max_depth=d, min_samples_leaf=m) for d in (3, 5, 8) for m in (10, 30)],
    'RandomForest': [dict(n_estimators=300, max_depth=d, min_samples_leaf=m) for d in (4, 8) for m in (10, 30)],
    'SVM':          [dict(C=C, gamma=g) for C in (0.1, 1, 10) for g in ('scale', 0.01)],
    'XGBoost':      [dict(n_estimators=200, max_depth=d, learning_rate=lr, subsample=0.8, colsample_bytree=0.8)
                     for d in (2, 4) for lr in (0.03, 0.1)],
}

pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))

def make_sequences(Xs, ys, a, b, L):
    lo = max(0, a - L + 1)
    Xb, yb = Xs[lo:b], ys[lo:b]
    Xseq = np.stack([Xb[k - L + 1:k + 1] for k in range(L - 1, len(Xb))]).astype('float32')
    return Xseq, yb[L - 1:]

# PyTorch LSTM architecture (used when TensorFlow is not installed)
if DL_BACKEND == 'pytorch':
    class TorchLSTM(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.lstm = nn.LSTM(in_features, 32, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(32, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 1)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            out, (hn, _) = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            out = self.relu(self.fc1(out))
            return self.sig(self.fc2(out))

    def train_torch_lstm(X_tr, y_tr, X_val, y_val, in_dim, epochs=40, batch_size=64, patience=5):
        torch.manual_seed(SEED)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TorchLSTM(in_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss(reduction='none')

        pos_w = 0.5 / max(1e-4, y_tr.mean())
        neg_w = 0.5 / max(1e-4, 1 - y_tr.mean())

        x_t = torch.tensor(X_tr, dtype=torch.float32)
        y_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(x_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        x_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        best_loss = float('inf')
        best_state = None
        no_imp = 0

        for epoch in range(epochs):
            model.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                w = torch.where(by == 1, pos_w, neg_w)
                loss = (criterion(out, by) * w).mean()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(x_v)
                v_loss = criterion(val_out, y_v).mean().item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

preds, val_bal, fitted, scalers, best_params = {}, {}, {}, {}, {}

print("Training models across 10 configurations...")
for fs, cols in FEATURE_SETS.items():
    scaler = StandardScaler().fit(Xtr[cols])  # Fitted strictly on TRAIN
    scalers[fs] = scaler
    A_tr = scaler.transform(Xtr[cols])
    A_va = scaler.transform(Xva[cols])
    A_te = scaler.transform(Xte[cols])

    for name, grid in GRID.items():
        best_p, best_s = None, -1
        for p in grid:
            m = make_model(name, p, pos_weight).fit(A_tr, ytr)
            s = balanced_accuracy_score(yva, m.predict(A_va))
            if s > best_s:
                best_p, best_s = p, s
        final_m = make_model(name, best_p, pos_weight).fit(A_tr, ytr)
        fitted[(fs, name)] = final_m
        best_params[(fs, name)] = best_p
        val_bal[(fs, name)] = best_s
        preds[(fs, name)] = final_m.predict(A_te)
        print(f"[{fs:9s}] {name:13s} -> Val Balanced Acc: {best_s:.3f} | Best: {best_p}")

    # LSTM Network (TensorFlow or PyTorch)
    Xs_all = scaler.transform(X[cols])
    y_all = y.values
    Xs_tr, ys_tr = make_sequences(Xs_all, y_all, 0, i1, SEQ_LEN)
    Xs_va, ys_va = make_sequences(Xs_all, y_all, i1, i2, SEQ_LEN)
    Xs_te, ys_te = make_sequences(Xs_all, y_all, i2, n, SEQ_LEN)

    if DL_BACKEND == 'tensorflow':
        tf.keras.utils.set_random_seed(SEED)
        lstm = models.Sequential([
            layers.Input(shape=(SEQ_LEN, len(cols))),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')
        cw = {0: 0.5 / max(1e-4, (1 - ytr.mean())), 1: 0.5 / max(1e-4, ytr.mean())}
        lstm.fit(
            Xs_tr, ys_tr, validation_data=(Xs_va, ys_va),
            epochs=40, batch_size=64, class_weight=cw, verbose=0,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        va_preds = (lstm.predict(Xs_va, verbose=0).ravel() > 0.5).astype(int)
        te_preds = (lstm.predict(Xs_te, verbose=0).ravel() > 0.5).astype(int)
        val_bal[(fs, 'LSTM')] = balanced_accuracy_score(ys_va, va_preds)
        preds[(fs, 'LSTM')] = te_preds
        fitted[(fs, 'LSTM')] = lstm
    else:
        lstm = train_torch_lstm(Xs_tr, ys_tr, Xs_va, ys_va, len(cols))
        with torch.no_grad():
            va_p = lstm(torch.tensor(Xs_va, dtype=torch.float32)).cpu().numpy().ravel()
            te_p = lstm(torch.tensor(Xs_te, dtype=torch.float32)).cpu().numpy().ravel()
        val_bal[(fs, 'LSTM')] = balanced_accuracy_score(ys_va, (va_p > 0.5).astype(int))
        preds[(fs, 'LSTM')] = (te_p > 0.5).astype(int)
        fitted[(fs, 'LSTM')] = lstm

    print(f"[{fs:9s}] {'LSTM':13s} -> Val Balanced Acc: {val_bal[(fs, 'LSTM')]:.3f}")

# ---------------------------------------------------------
# 6. Single-Pass Test Set Evaluation
# ---------------------------------------------------------
yt = yte.values
rows, cm_text = [], []

for (fs, name), p in preds.items():
    assert len(p) == len(yt), f"Length mismatch for {fs}/{name}"
    acc = accuracy_score(yt, p)
    prec = precision_score(yt, p, zero_division=0)
    rec  = recall_score(yt, p, zero_division=0)
    f1   = f1_score(yt, p, zero_division=0)
    ci95 = 1.96 * np.sqrt(acc * (1 - acc) / len(yt))
    vs_b = acc - baseline
    pval = binomtest(int((p == yt).sum()), len(yt), baseline, alternative='greater').pvalue

    rows.append(dict(
        feature_set=fs, model=name, accuracy=acc,
        precision=prec, recall=rec, f1=f1, pct_predicted_UP=p.mean(),
        ci95_plusminus=ci95, vs_baseline=vs_b, p_vs_baseline=pval
    ))
    tn, fp, fn, tp = confusion_matrix(yt, p, labels=[0, 1]).ravel()
    cm_text.append(
        f"{fs}_{name}\n"
        f"  actual DOWN: predicted DOWN={tn}, predicted UP={fp}\n"
        f"  actual UP:   predicted DOWN={fn}, predicted UP={tp}\n"
    )

res = pd.DataFrame(rows).set_index(['feature_set', 'model']).round(3)

for p in [OUT_DIR, '.']:
    res.to_csv(os.path.join(p, 'model_comparison_results.csv'))
    with open(os.path.join(p, 'confusion_matrices.txt'), 'w') as f:
        f.write('\n'.join(cm_text))

print("\n=== Model Comparison Results (Test Set) ===")
print(res)

# ---------------------------------------------------------
# 7. Model Comparison Chart
# ---------------------------------------------------------
order = ['DecisionTree', 'RandomForest', 'SVM', 'XGBoost', 'LSTM']
acc_tab = res['accuracy'].unstack('feature_set').loc[order]
ci_tab  = res['ci95_plusminus'].unstack('feature_set').loc[order]

fig, ax = plt.subplots(figsize=(9, 4.5))
acc_tab[['raw', 'indicator']].plot.bar(
    yerr=ci_tab[['raw', 'indicator']].T.values,
    capsize=3, ax=ax, color=['#8da0cb', '#4c72b0']
)
ax.axhline(baseline, color='red', ls=':', label=f'Majority Baseline ({baseline:.1%})')
ax.axhline(0.5, color='gray', ls='--', label='Random Chance (50%)')
ax.set_ylim(0.3, 0.7)
ax.set_ylabel('Test Accuracy')
ax.set_title(f'P-396: Next-Day UP/DOWN Accuracy ({TICKER})')
ax.legend(loc='lower right')
plt.xticks(rotation=0)
plt.tight_layout()

for p in [OUT_DIR, '.']:
    plt.savefig(os.path.join(p, 'model_comparison_chart.png'), dpi=150)
plt.close()

# ---------------------------------------------------------
# 8. Feature Importance
# ---------------------------------------------------------
fi_dict = {}
for nm in ['DecisionTree', 'RandomForest', 'XGBoost']:
    m = fitted[('indicator', nm)]
    fi_dict[nm] = pd.Series(m.feature_importances_, index=FEATURE_SETS['indicator'])

fi = pd.DataFrame(fi_dict)
fi = fi / fi.sum()
fi['average'] = fi.mean(axis=1)
fi = fi.sort_values('average', ascending=False).round(4)

for p in [OUT_DIR, '.']:
    fi.to_csv(os.path.join(p, 'feature_importances.csv'))

fig, ax = plt.subplots(figsize=(8, 4.5))
fi['average'].plot.bar(ax=ax, color='#4c72b0')
ax.set_ylabel('Normalized Mean Importance')
ax.set_title('Feature Importance (DecisionTree, RandomForest, XGBoost Average)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

for p in [OUT_DIR, '.']:
    plt.savefig(os.path.join(p, 'feature_importance_chart.png'), dpi=150)
plt.close()

print("\n=== Feature Importances (Top 6) ===")
print(fi.head(6))

# ---------------------------------------------------------
# 9. Pipeline Integrity Assertions
# ---------------------------------------------------------
rng = np.random.default_rng(SEED)
for i in rng.integers(0, len(data) - 1, 20):
    d = data.index[i]
    pos = raw_prices.index.get_loc(d)
    assert (raw_prices['Close'].iloc[pos + 1] > raw_prices['Close'].iloc[pos]) == bool(data.loc[d, 'target'])

assert ind_df['SMA_10'].iloc[WARMUP] == raw_prices['Close'].iloc[WARMUP-9:WARMUP+1].mean()
assert Xtr.index.max() < Xva.index.min() < Xva.index.max() < Xte.index.min()
print("\nAll 6 Pipeline Integrity & Non-Leakage Assertions Passed.")

# ---------------------------------------------------------
# 10. Live Next-Day Forward Inference
# ---------------------------------------------------------
best_key = max(val_bal, key=val_bal.get)
fs, name = best_key
cols = FEATURE_SETS[fs]
last_row = ind_df.iloc[-1:][cols]

if name == 'LSTM':
    win = ind_df.iloc[-SEQ_LEN:][cols]
    if DL_BACKEND == 'tensorflow':
        p_up = float(fitted[best_key].predict(scalers[fs].transform(win).astype('float32')[None], verbose=0)[0, 0])
    else:
        with torch.no_grad():
            w_t = torch.tensor(scalers[fs].transform(win).astype('float32')[None])
            p_up = float(fitted[best_key](w_t).item())
else:
    p_up = float(fitted[best_key].predict_proba(scalers[fs].transform(last_row))[0, 1])

final_call = {
    'last_data_date': str(raw_prices.index[-1].date()),
    'model': name,
    'feature_set': fs,
    'p_up': round(p_up, 3),
    'prediction': 'UP' if p_up > 0.5 else 'DOWN'
}

for p in [OUT_DIR, '.']:
    with open(os.path.join(p, 'final_output.json'), 'w') as f:
        json.dump(final_call, f, indent=2)

print("\n=== Live Next-Day Signal ===")
print(f"Data Date: {final_call['last_data_date']}")
print(f"Model: {final_call['model']} ({final_call['feature_set']}) -> Call: {final_call['prediction']} (P(UP) = {final_call['p_up']})")
print("Pipeline run complete. All outputs saved to outputs/ and root.")

