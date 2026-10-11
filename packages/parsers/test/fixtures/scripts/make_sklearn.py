"""Generate scikit-learn fixtures (pickle + joblib dumps of small fitted models).

Requires: scikit-learn, joblib (bundled with scikit-learn).
Run from anywhere: python make_sklearn.py

Models (all fitted on the iris dataset, 4 features / 3 classes, seed 0):
  pipeline_logreg   Pipeline(StandardScaler, LogisticRegression)
  mlp               MLPClassifier(hidden_layer_sizes=(16, 8))
  forest            RandomForestClassifier(n_estimators=3, max_depth=3)
  column_pipeline   Pipeline(ColumnTransformer(StandardScaler | PCA(2)), LogisticRegression)
Files: <name>.pkl (pickle protocol 4), <name>.joblib (uncompressed),
plus pipeline_logreg_p5.pkl (protocol 5) and pipeline_logreg_z.joblib (zlib, compress=3).
"""
import os
import pickle

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = os.path.join(os.path.dirname(__file__), "..", "sklearn")
os.makedirs(OUT, exist_ok=True)

X, y = load_iris(return_X_y=True)
models = {
    "pipeline_logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]),
    "mlp": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=0),
    "forest": RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0),
    "column_pipeline": Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), [0, 1]), ("pca", PCA(n_components=2), [2, 3])])),
        ("clf", LogisticRegression(max_iter=500)),
    ]),
}
import warnings
warnings.filterwarnings("ignore")
for name, m in models.items():
    m.fit(X, y)
    with open(os.path.join(OUT, f"{name}.pkl"), "wb") as f:
        pickle.dump(m, f, protocol=4)
    joblib.dump(m, os.path.join(OUT, f"{name}.joblib"))
with open(os.path.join(OUT, "pipeline_logreg_p5.pkl"), "wb") as f:
    pickle.dump(models["pipeline_logreg"], f, protocol=5)
joblib.dump(models["pipeline_logreg"], os.path.join(OUT, "pipeline_logreg_z.joblib"), compress=3)
print("wrote", sorted(os.listdir(OUT)))
