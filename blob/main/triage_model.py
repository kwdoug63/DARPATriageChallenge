"""
TA2 Applicant Challenge — Age Group Classification from Arterial Pressure Waveforms
Team: Sober_Agents
Author: Kenneth Wayne Douglas, MD
Contact: kenneth.douglas@soberagents.ai

Model: Ensemble (ExtraTrees + RandomForest) on engineered physiological features
       + PCA-compressed waveform representation
Target: 6-class age group prediction (20s=0, 30s=1, 40s=2, 50s=3, 60s=4, 70s=5)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
aorta_train = pd.read_csv('aortaP_train_data.csv', index_col=0)
brach_train  = pd.read_csv('brachP_train_data.csv', index_col=0)
aorta_test   = pd.read_csv('aortaP_test_data.csv',  index_col=0)
brach_test   = pd.read_csv('brachP_test_data.csv',  index_col=0)

y_train   = aorta_train['target'].values
aorta_tr  = np.nan_to_num(aorta_train.drop(columns=['target']).values)
brach_tr  = np.nan_to_num(brach_train.drop(columns=['target']).values)
aorta_te  = np.nan_to_num(aorta_test.values)
brach_te  = np.nan_to_num(brach_test.values)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def extract_features(aorta: np.ndarray, brach: np.ndarray) -> np.ndarray:
    """
    Extract physiologically motivated features from paired aortic/brachial
    pressure waveforms sampled at 500 Hz (336 samples per waveform).

    Per-signal features (computed for aorta and brachial independently):
      - Statistical moments: mean, std, min, max, p10, p90, skewness, kurtosis
      - Pulse pressure (PP = systolic - diastolic proxy)
      - Mean arterial pressure proxy (MAP = diastolic + PP/3)
      - First derivative (dP/dt): max, min, mean, std
        → dP/dt max is a validated marker of arterial stiffness that increases with age
      - FFT spectral power in three bands (low/mid/high frequency)
      - Waveform energy (sum of squared FFT coefficients)
      - Area under the waveform (trapezoid integration)
      - Systolic peak count and mean peak amplitude
      - Temporal thirds means (early, mid, late phase of the cardiac cycle)

    Cross-signal features (brachial vs. aortic):
      - Brachial-aortic pressure amplification (mean, systolic, min, std)
        → Peripheral amplification decreases with age due to arterial stiffening
      - Waveform correlation coefficient
      - Difference signal statistics (mean, std)
    """
    n = aorta.shape[0]
    features = []

    for i in range(n):
        a, b = aorta[i], brach[i]
        row = []

        for sig in [a, b]:
            d    = np.diff(sig)
            fft  = np.abs(np.fft.rfft(sig))
            pp   = sig.max() - sig.min()
            map_ = sig.min() + pp / 3.0
            pks, _ = find_peaks(sig, prominence=2)

            row += [
                # Statistical
                sig.mean(), sig.std(), sig.min(), sig.max(), pp, map_,
                float(np.percentile(sig, 10)), float(np.percentile(sig, 90)),
                float(stats.skew(sig)), float(stats.kurtosis(sig)),
                # Rate of change
                d.max(), d.min(), d.mean(), d.std(),
                # Spectral
                float(fft[1:5].mean()), float(fft[5:15].mean()), float(fft[15:].mean()),
                float(np.sum(fft**2)),
                # Integral
                float(np.trapezoid(sig)),
                # Peak features
                len(pks),
                float(sig[pks].mean()) if len(pks) > 0 else float(sig.mean()),
                # Temporal thirds
                float(sig[:112].mean()),
                float(sig[112:224].mean()),
                float(sig[224:].mean()),
            ]

        # Cross-signal features
        ds = b - a
        row += [
            float(b.mean() - a.mean()),
            float(b.max()  - a.max()),
            float(b.min()  - a.min()),
            float(b.std()  - a.std()),
            float(np.corrcoef(a, b)[0, 1]),
            float(ds.mean()), float(ds.std()),
        ]
        features.append(row)

    return np.nan_to_num(np.array(features, dtype=np.float64))


print("Extracting features...")
X_eng_tr = extract_features(aorta_tr, brach_tr)
X_eng_te = extract_features(aorta_te, brach_te)
print(f"  Engineered feature matrix: {X_eng_tr.shape}")

# ─────────────────────────────────────────────
# 3. PCA WAVEFORM COMPRESSION
# ─────────────────────────────────────────────
# Compress the raw waveforms (672-dim = 336 aorta + 336 brachial) to 50 PCs
# capturing ~54% of total variance — encodes waveform shape information
# complementary to the hand-crafted features above.

both_tr = np.hstack([aorta_tr, brach_tr])
both_te = np.hstack([aorta_te, brach_te])

scaler_wave = StandardScaler()
both_tr_sc  = scaler_wave.fit_transform(both_tr)
both_te_sc  = scaler_wave.transform(both_te)

pca = PCA(n_components=50, random_state=42)
pca_tr = pca.fit_transform(both_tr_sc)
pca_te = pca.transform(both_te_sc)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

X_tr = np.hstack([X_eng_tr, pca_tr])
X_te = np.hstack([X_eng_te, pca_te])

# Final scaling
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

# ─────────────────────────────────────────────
# 4. ENSEMBLE MODEL
# ─────────────────────────────────────────────
# Soft-voting ensemble of ExtraTreesClassifier + RandomForestClassifier.
# Both models use bagging with random feature subsets (30-40% per split),
# which reduces variance on this moderately high-dimensional feature space.

print("Training ensemble...")
et = ExtraTreesClassifier(
    n_estimators=400,
    max_features=0.4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf = RandomForestClassifier(
    n_estimators=400,
    max_features=0.3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

et.fit(X_tr_sc, y_train)
rf.fit(X_tr_sc, y_train)

# Soft vote: average class probabilities
proba = (et.predict_proba(X_te_sc) + rf.predict_proba(X_te_sc)) / 2
preds = np.argmax(proba, axis=1)

# ─────────────────────────────────────────────
# 5. OUTPUT
# ─────────────────────────────────────────────
output = {int(idx): int(p) for idx, p in zip(aorta_test.index, preds)}

print(f"\nPrediction distribution:")
for cls, cnt in sorted(pd.Series(preds).value_counts().sort_index().items()):
    age = ['20s','30s','40s','50s','60s','70s'][cls]
    print(f"  Class {cls} ({age}): {cnt}")

with open('Sober_Agents_output.json', 'w') as f:
    json.dump(output, f, indent=4)

print("\nSaved: Sober_Agents_output.json ✓")
