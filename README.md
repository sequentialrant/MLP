# Multi-layer Perceptron (MLP)
This repository contains NumPy implementation of multi-layer perceptron.

# Setup Instructions
```bash
cd src
pip install -U requirements.txt
```

# How to Reproduce
1. Create a Jupyter lab file
```bash
jupyter lab Experiment.ipynb
```
2. Load libraries
```python
from src.utils import split_data
from src.utils import standardize_data
from src.utils import performance_metrics
from src.nn import NeuralNetworkClassifier
from src.utils import visualize_decision_boundary
```
3. Load Data
```python
data = np.load("data/processed/processed_data.npz")
X_test = data['X_test']
y_test = data['y_test']
```
4. Load Model and Predict
```python
import joblib
clf = joblib.load("models/mlp.sav")
y_prob = clf.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
```
5. Visualize Decision Boundry
```python
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.title("Train")
visualize_decision_boundary(model=clf, X=X_train, y=y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
visualize_decision_boundary(model=clf, X=X_test, y=y_test)

plt.show()
```

