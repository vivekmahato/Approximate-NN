# Imports
### Importing all the necessary libraries and our MFModels code file.
```python
import numpy as np
from MFModels import AnnoyClassifier, HNSWClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
```


# Load Data
## Loading data and splitting it into train and test sets.
```python
data = np.load("data/plarge300.npy",allow_pickle=True).item()
X_train, X_test, y_train, y_test = train_test_split(data["X"],data["y"], test_size=0.5, random_state=1992)
```

# Parameter Search w/o Mac-Fac Strategy
## We would be using GridSearchCV from sklearn to find the best set of parameters for the model on the train set.
```python
param_grid = {
    "n_neighbors": np.arange(1, 11, 2)
}
print(param_grid)

annoy = AnnoyClassifier(random_seed=1992)
gscv = GridSearchCV(annoy, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
gscv = gscv.fit(X_train,y_train)

best_param = gscv.best_params_
best_score = gscv.best_score_

print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)
```

    Best Parameters:  {'n_neighbors': 7}
    Best Accuracy:  0.82


# Model Evaluation
## Supply the best set of parameters to the Annoy model: train with Train set, and test on held-out test set.
```python
annoy = AnnoyClassifier(**best_param,random_seed=1992).fit(X_train,y_train)
y_hat = annoy.predict(X_test)
acc = accuracy_score(y_test,y_hat)
print("Model accuracy w/o Mac-Fac: ", round(acc, 2))
```

    Model accuracy w/o Mac-Fac:  0.78


# Parameter Search w Mac-Fac Strategy
## We would be using GridSearchCV from sklearn to find the best set of parameters for the model and DTW on the train set.
```python
radii = np.arange(1,11,2)
param_grid = {
    "n_neighbors": np.arange(1, 11, 2),
    "mac_neighbors": np.arange(10, 50, 5),
    'sakoe_chiba_radius': np.arange(1, 11, 2)

}
```


```python
annoy = AnnoyClassifier(random_seed=1992)
gscv = GridSearchCV(annoy, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
gscv = gscv.fit(X_train, y_train)

best_param = gscv.best_params_
best_score = gscv.best_score_

print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)
```

    Best Parameters:  {'mac_neighbors': 30, 'n_neighbors': 7, 'sakoe_chiba_radius': 9}
    Best Accuracy:  0.8333333333333333


# Model Evaluation
## Supply the best set of parameters to the Annoy model: train with Train set, and test on held-out test set.
```python
annoy = AnnoyClassifier(**best_param,random_seed=1992).fit(X_train,y_train)
y_hat = annoy.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Model accuracy w/ Mac-Fac: ", round(acc, 2))
```

    Model accuracy w/ Mac-Fac:  0.8



```python


```
