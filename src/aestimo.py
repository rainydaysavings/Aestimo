import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT'))


def model_classification(train, model_type='rf', **kwargs):
    """
    Trains a classification model on the given training data and returns the feature importances.

    Parameters:
        train (DataFrame): The training data containing the features and labels.
        model_type (str): The type of classification model to use. Default is 'rf' (Random Forest).
        **kwargs: Additional keyword arguments to be passed to the classification model.

    Returns:
        dict: A dictionary containing the feature importances, where the keys are the feature names and the values are the importances.

    Raises:
        ValueError: If an invalid model type is provided.
    """

    train = train.copy()
    x_train = train.drop('label', axis=1).select_dtypes(include=[np.number])
    y_train = train['label']

    if model_type == 'rf':
        model = RandomForestClassifier(**kwargs)
    elif model_type == 'lda':
        model = LDA(**kwargs)
    else:
        raise ValueError("Invalid model type")

    model.fit(x_train, y_train)

    if model_type == 'rf':
        rf_importances = model.feature_importances_
        return {k: v for k, v in zip(x_train.columns, rf_importances)}

    if model_type == 'lda':
        lda_importances = np.abs(model.coef_[0])
        return {k: v for k, v in zip(x_train.columns, lda_importances)}


def feature_correlation(train, feature_cols):
    """
    Generates a correlation heatmap of specified feature variables in the given dataset.

    Args:
        train (pd.DataFrame): The input dataset.
        feature_cols (List[str]): The list of feature variables to calculate correlation for.

    Returns:
        pd.DataFrame: The correlation matrix of the specified feature variables.
    """

    train = train.copy()

    # Scale only the specified feature variables
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train[feature_cols])

    # Create a DataFrame from scaled_features for correlation analysis
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
    correlation_matrix = scaled_df.corr()

    plt.figure(figsize=(16, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    return correlation_matrix


def pca_weighted(train, features_for_pca, importances):
    """
    Applies weighted PCA on the given training data.

    Args:
        train (pd.DataFrame): The training dataset.
        features_for_pca (List[str]): List of feature names to be used for PCA.
        importances (Dict[str, float]): Dictionary of feature importances.

    Returns:
        Tuple[pd.DataFrame, float]: A tuple containing the modified training dataset and the correlation
        of the new feature with the target label.
    """

    train = train.copy()

    # Select and scale the features
    x_for_pca = train[list(features_for_pca)]
    scaler = StandardScaler()
    y_train = train['label']
    x_for_pca_scaled = scaler.fit_transform(x_for_pca, y_train)

    # Apply square root of importance weights
    for feature in features_for_pca:
        x_for_pca_scaled[:, list(features_for_pca).index(feature)] *= np.sqrt(importances[feature])

    # Apply PCA and keep only the first principal component
    pca = PCA(n_components=1)
    x_pca_result = pca.fit_transform(x_for_pca_scaled)

    # Create a unique column name for this pair to avoid overwriting
    col_name = f"Weighted_PCA_feature_{'_'.join(features_for_pca)}"
    train[col_name] = x_pca_result

    # Calculate the correlation of this new feature with the target label
    new_corr_matrix_pca = train.corr()

    return train, new_corr_matrix_pca.loc[col_name, 'label']


def get_pca_correlation_map(train, importances):
    """
    Calculates the correlation map between pairs of features using PCA.

    Args:
        train (pd.DataFrame): The training dataset.
        importances (dict): The importances of each feature.

    Returns:
        Tuple[Dict[Tuple[str, str], float], List[Tuple[str, str]]]: A tuple containing the correlation map and the list
        of valid feature pairs.
    """

    correlation_map = {}

    # Get all unique pairs of features
    if 'label' in train:
        train.drop('label', axis=1)

    # From doing Cumulative Distribution analysis of features pairs (see: cdf_plotter.py)
    valid_feature_pairs = [
        ('acousticness', 'danceability'), ('acousticness', 'energy'), ('acousticness', 'liveness'),
        ('acousticness', 'speechiness'),
        ('danceability', 'liveness'), ('danceability', 'speechiness'), ('danceability', 'speechiness'),
        ('energy', 'liveness'), ('energy', 'speechiness'), ('energy', 'valence'),
        ('instrumentalness', 'valence'),
        ('speechiness', 'liveness'),
        ('liveness', 'valence'),
    ]

    # Create a copy of the original DataFrame to avoid inplace modifications
    train_copy = train.copy()
    for pair in valid_feature_pairs:
        # Apply the PCA
        train, corr_value = pca_weighted(train_copy, pair, importances)
        # Add to correlation map
        correlation_map[pair] = corr_value

    return correlation_map, valid_feature_pairs


# Plot a bar chart for feature pairs based on their correlation with target.
def plot_correlation_map(correlation_map):
    """
    Plots a correlation map of feature pairs with their correlations.

    Args:
        correlation_map (dict): A dictionary containing feature pairs as keys and their correlations as values.

    Returns:
        None

    The function converts the correlation map to a list of tuples and sorts them by correlation in descending order.
    It then extracts the features and correlations from the sorted list.
    The feature tuples are converted to strings for plotting purposes.
    The function creates a bar plot with the feature strings on the y-axis and the correlations on the x-axis.
    The plot is displayed with the x-axis labeled as 'Correlation with Target', y-axis as 'Feature Pairs',
    and the title as 'Correlation of PCA Feature Pairs with Target'.
    """

    # Convert the correlation map to a list of tuples and sort by correlation
    sorted_correlations = sorted(correlation_map.items(), key=lambda x: x[1], reverse=True)
    features, correlations = zip(*sorted_correlations)

    # Convert the feature tuples to strings for plotting
    feature_strings = [f"{f1} + {f2}" for f1, f2 in features]
    plt.figure(figsize=(16, 16))
    plt.barh(feature_strings, correlations, color='blue')
    plt.xlabel('Correlation with Target')
    plt.ylabel('Feature Pairs')
    plt.title('Correlation of PCA Feature Pairs with Target')
    plt.show()


def plot_importances(importances, title):
    """
    Plots the importances of features in a bar chart.

    Args:
        importances (dict): A dictionary containing feature names as keys and their corresponding importances as values.
        title (str): The title of the plot.

    Returns:
        None: This function does not return anything, it only plots the bar chart.
    """

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])

    # Sort the DataFrame by the importances
    df = df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(16, 16))
    sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
    plt.title(title)
    plt.show()


def mutual_information(train):
    """
    Measures the degree of dependence between parameters pairs.

    Args:
        train (DataFrame): The loaded training DataFrame
    Return:
        The measured degree of dependence
    """

    X = train.drop('label', axis=1)
    y = train['label']

    mi = mutual_info_classif(X, y)
    mi_series = pd.Series(mi, index=X.columns)
    mi_series.sort_values(ascending=False, inplace=True)

    return mi


def information_gain(train):
    """
    Measures the reduction in entropy after splitting a dataset along a particular attribute or feature. A decision tree
    algorithm uses information gain to determine what attributes or features contribute most towards classifying
    instances correctly. Higher information gain means better classification accuracy at the leaf node level.

    Args:
        train: The loaded training DataFrame
    Return:
        The measured reduction in entropy
    """

    X = train.drop('label', axis=1)
    y = train['label']

    tree_clf = DecisionTreeClassifier(criterion='entropy')
    tree_clf.fit(X, y)

    ig = tree_clf.feature_importances_
    ig_series = pd.Series(ig, index=X.columns)
    ig_series.sort_values(ascending=False, inplace=True)

    return ig


def create_composite_features(train, test):
    """
    Creates composite features based on mutual information (MI), information gain (IG), and random forest feature importances.

    Args:
        train (DataFrame): The training data.
        test (DataFrame): The test data.

    Returns:
        tuple: A tuple containing the modified training data and test data.
    """

    train = train.copy()
    test = test.copy()

    # 1. Compute MI and IG and normalize
    mi_scores = mutual_information(train)
    mi_sum = np.sum(mi_scores)
    normalized_mi_scores = {k: v / mi_sum for k, v in zip(train.drop('label', axis=1).columns, mi_scores)}

    ig_scores = information_gain(train)
    ig_sum = np.sum(ig_scores)
    normalized_ig_scores = {k: v / ig_sum for k, v in zip(train.drop('label', axis=1).columns, ig_scores)}

    # Initial model to get Random Forest feature importances
    rf_importances = model_classification(train, 'rf', n_estimators=50, random_state=42)

    # 2. Compute composite scores
    features = train.drop('label', axis=1).columns
    composite_scores = {
        k: (normalized_mi_scores.get(k, 0) + normalized_ig_scores.get(k, 0) + rf_importances.get(k, 0)) / 3
        for k in features}

    # This block will execute for both training and test data
    # Get updated feature list
    updated_features = train.columns.tolist()
    if 'label' in updated_features:
        updated_features.remove('label')

    # 4. Create composite features
    correlation_map, top_pairs = get_pca_correlation_map(train, composite_scores)
    valid_combinations = [comb for comb in top_pairs if all(x in train.columns for x in comb)]
    for comb in valid_combinations:
        feature_name = '+'.join(comb)
        train[feature_name] = train[list(comb)].mean(axis=1)
        test[feature_name] = test[list(comb)].mean(axis=1)

    return train, test


def get_top_features(ensemble_score, correlation_map, top_n=20):
    """
    Returns a list of top features based on ensemble scores and correlation map.

    Args:
        ensemble_score (dict): A dictionary containing feature scores from an ensemble model.
        correlation_map (dict): A dictionary containing feature correlation values.
        top_n (int, optional): The number of top features to return. Defaults to 20.

    Returns:
        list: A list of top features based on ensemble scores and correlation map.
    """

    top_importance_features = sorted(ensemble_score, key=ensemble_score.get, reverse=True)[:top_n]
    top_correlation_features = {f for pair, _ in
                                sorted(correlation_map.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n] for f in
                                pair}
    return list(set(top_importance_features) | top_correlation_features)


def print_scores(clf, X, y):
    """
    Prints the average accuracy, precision, and recall scores of a classifier.
    Stratified K-fold scoring by mean, shown to better approx. than K-fold over competition results metrics.

    Args:
        clf (estimator): The classifier to evaluate.
        X (array-like): The input features.
        y (array-like): The target variable.

    Returns:
        None
    """

    cv = StratifiedKFold()

    scores = cross_validate(estimator=clf,
                            X=X, y=y,
                            scoring=['accuracy', 'precision', 'recall'],
                            cv=cv)

    print("Accuracy:", round(scores["test_accuracy"].mean(), 3))
    print("Precision:", round(scores["test_precision"].mean(), 3))
    print("Recall:", round(scores["test_recall"].mean(), 3))


def predict_with_meta_classifier(train, test, ensemble_score, correlation_map):
    """
    Trains with training data, and predicts the labels for the training and test data using a meta-classifier of
    classifiers SVM, and GBM.

    Args:
        train (DataFrame): The training data.
        test (DataFrame): The test data.
        ensemble_score (array-like): The ensemble scores for each feature.
        correlation_map (array-like): The correlation map for the features.

    Returns:
        tuple: A tuple containing the predicted labels for the training data and the test data.

    Raises:
        None
    """

    selected_features = get_top_features(ensemble_score, correlation_map)

    # Split and scale the training data
    X_train = train[selected_features]
    y_train = train['label']
    X_val, _, y_val, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train GBM
    gbm_clf = GradientBoostingClassifier(n_estimators=50)
    gbm_clf.fit(X_train_scaled, y_train)
    gbm_predictions = gbm_clf.predict(X_val_scaled)

    # Train and predict SVM
    svm_clf = SVC(C=1.0, kernel='poly', degree=3, gamma='auto', coef0=0.5)
    svm_clf.fit(X_train_scaled, y_train)
    svm_predictions = svm_clf.predict(X_val)

    # Generate meta-features
    meta_features = np.column_stack((gbm_predictions, svm_predictions))

    # Train a meta-classifier
    meta_clf = LogisticRegression()
    meta_clf.fit(meta_features, y_val)

    # Score training test
    train_predictions = meta_clf.predict(meta_features)
    print("[GBM classifier score]")
    print_scores(gbm_clf, X_train_scaled, y_train)
    print("\n")
    print("[SVM classifier score]")
    print_scores(svm_clf, X_train_scaled, y_train)
    print("\n")
    print("[GBM+SVM meta-classifier score]")
    print_scores(meta_clf, meta_features, y_val)
    print("\n")

    # Test Predictions
    X_test = test[selected_features]
    X_test_scaled = scaler.transform(X_test)  # Use the scaler fitted on the training data

    gbm_test_predictions = gbm_clf.predict(X_test_scaled)
    svm_test_predictions = svm_clf.predict(X_test_scaled)
    test_meta_features = np.column_stack((gbm_test_predictions, svm_test_predictions))
    test_predictions = meta_clf.predict(test_meta_features)
    return train_predictions, test_predictions


def create_ensemble_score(train, alpha=0.5, beta=0.5):
    """
    Calculates the ensemble score for each feature based on normalized mutual information, normalized information gain,
    and feature importances from Random Forest and LDA models.

    Args:
        train (DataFrame): The training dataset.
        alpha (float, optional): The weight for the feature importances from Random Forest and LDA models.
        Defaults to 0.5.
        beta (float, optional): The weight for the normalized mutual information and normalized information gain.
        Defaults to 0.5.

    Returns:
        dict: A dictionary containing the ensemble score for each feature.
    """

    # Get mutual information scores
    mi_scores = mutual_information(train)
    mi_sum = np.sum(mi_scores)
    normalized_mi_scores = {k: v / mi_sum for k, v in zip(train.drop('label', axis=1).columns, mi_scores)}
    mi_df = pd.DataFrame(list(normalized_mi_scores.items()), columns=['Feature', 'Normalized_MI'])
    mi_df = mi_df.sort_values('Normalized_MI', ascending=False)
    plt.figure(figsize=(16, 16))
    plt.barh(mi_df['Feature'], mi_df['Normalized_MI'], color='skyblue')
    plt.xlabel('Normalized Mutual Information')
    plt.ylabel('Feature')
    plt.title('Feature Importance based on Mutual Information')
    plt.show()

    # Get information gain scores
    ig_scores = information_gain(train)
    ig_sum = np.sum(ig_scores)
    normalized_ig_scores = {k: v / ig_sum for k, v in zip(train.drop('label', axis=1).columns, ig_scores)}
    ig_df = pd.DataFrame(list(normalized_ig_scores.items()), columns=['Feature', 'Normalized_IG'])
    ig_df = ig_df.sort_values('Normalized_IG', ascending=False)
    plt.figure(figsize=(16, 16))
    plt.barh(ig_df['Feature'], ig_df['Normalized_IG'], color='skyblue')
    plt.xlabel('Normalized Information Gain')
    plt.ylabel('Feature')
    plt.title('Feature Importance based on Information Gain')
    plt.show()

    # Get random forest feature importance's
    # Random Forest
    rf_importances = model_classification(train, model_type='rf', n_estimators=50, random_state=42)
    plot_importances(rf_importances, 'Feature Importances from Random Forest')
    rf_sum = sum(rf_importances.values())
    normalized_rf_importances = {k: v / rf_sum for k, v in rf_importances.items()}

    # LDA
    lda_importances = model_classification(train, model_type='lda', solver='svd')
    plot_importances(lda_importances, 'Feature Importances from LDA')

    # Normalize lda_importances (similar to how rf_importances was normalized)
    lda_sum = sum(lda_importances.values())
    normalized_lda_importances = {k: v / lda_sum for k, v in lda_importances.items()}

    # Create ensemble scores (now includes lda_importances)
    ensemble_score = {
        feature: alpha * (normalized_rf_importances.get(feature, 0) + normalized_lda_importances.get(feature, 0)) / 2 +
                 beta * (normalized_mi_scores.get(feature, 0) + normalized_ig_scores.get(feature, 0)) / 2
        for feature in set(normalized_rf_importances.keys()) | set(normalized_lda_importances.keys()) | set(
            normalized_mi_scores.keys()) | set(normalized_ig_scores.keys())}

    return ensemble_score


def main():
    # Load data
    train = pd.read_csv(PROJECT_ROOT / 'data/training_data.csv')
    test = pd.read_csv(PROJECT_ROOT / 'data/songs_to_classify.csv')

    # Pre-Process
    train_with_composites, test_with_composites = create_composite_features(train, test)
    ensemble_score = create_ensemble_score(train_with_composites)
    correlation_map, top_pairs = get_pca_correlation_map(train_with_composites.copy(), ensemble_score.copy())
    feature_correlation(train_with_composites.copy(), ensemble_score.copy())

    # Training
    train_predictions, test_predictions = predict_with_meta_classifier(train_with_composites.copy(),
                                                                       test_with_composites.copy(),
                                                                       ensemble_score.copy(), correlation_map.copy())
    print("Predictions on training data: ", train_predictions)
    print("\n")

    # Print predictions on test data
    print("Predictions on new data (not trained upon!): ", test_predictions)


if __name__ == "__main__":
    main()
