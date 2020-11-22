import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from MFModels import AnnoyClassifier

data = np.load("data/plarge300.npy", allow_pickle=True).item()
X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.5, random_state=1992)

param_grid = {
    "n_neighbors": np.arange(1, 11, 2)
}
annoy = AnnoyClassifier(random_seed=1992)
gscv = GridSearchCV(annoy, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
gscv = gscv.fit(X_train, y_train)

best_param = gscv.best_params_
best_score = gscv.best_score_

print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)

annoy = AnnoyClassifier(**best_param, random_seed=1992).fit(X_train, y_train)
y_hat = annoy.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Model accuracy w/o Mac-Fac: ", round(acc, 2))

radii = np.arange(1, 11, 2)
param_grid = {
    "n_neighbors": np.arange(1, 11, 2),
    "mac_neighbors": np.arange(10, 50, 5),
    'sakoe_chiba_radius': np.arange(1, 11, 2)

}

annoy = AnnoyClassifier(random_seed=1992)
gscv = GridSearchCV(annoy, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
gscv = gscv.fit(X_train, y_train)

best_param = gscv.best_params_
best_score = gscv.best_score_

print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)

annoy = AnnoyClassifier(**best_param, random_seed=1992).fit(X_train, y_train)
y_hat = annoy.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Model accuracy w/ Mac-Fac: ", round(acc, 2))
