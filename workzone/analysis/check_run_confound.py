import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

f1 = np.load("data/workzone_complex/features.npy")
f2 = np.load("data/workzone_complex_run2/features.npy")

X = np.concatenate([f1[:, 50, :], f2[:, 50, :]])
y = np.concatenate([np.zeros(len(f1)), np.ones(len(f2))])

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
print(f"run-identity AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
