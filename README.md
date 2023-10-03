# Aestimo
Music preference prediction model based on SVM and GBM

## Features
- Cumulative Distribution analysis of features pairs
- PCA used to create new features based on combinations of existing ones, maximizing the variance (some existing features are highly correlated)
- LDA to complement PCA by focusing on class separability
- Random Forests for feature importance estimation, to weigh the features before applying PCA, and as part of the ensemble score of features
- Mutual Information and Information Gain taken into account for the aforementioned ensemble score
- Meta-classifier based on SVM and GBM

## Steps to run code
```
$ python3 -m venv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
$ PROJECT_ROOT=your_project_root_location python3 src/songify.py
```

_Note: get_pca_correlation_map(train, importances) has a feature pair array specific to the training data, this was obtained by doing prior CDF analysis. Change values according to your own data, the rest should be generalized with the exception of training and test data filenames_
