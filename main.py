
import os

import numpy as np
from scipy.io import loadmat
from scipy.stats import poisson
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt


plt.rcParams["figure.figsize"] = (8, 6)


class NaiveBayesClassifier:

    def __init__(self):
        self.n_classes = None
        self.n_features = None
        self.classes_features_mean = None

    def fit(self, classes_samples_features):
        self.n_classes = len(classes_samples_features)
        self.n_features = len(classes_samples_features[0][0])
        self.classes_features_mean = np.empty(shape=(self.n_classes, self.n_features))
        for class_idx, class_samples_features in enumerate(classes_samples_features):
            self.classes_features_mean[class_idx] = np.mean(class_samples_features, axis=0)

    def predict(self, samples_features):
        classes_samples_features_logprob = poisson.logpmf(
            samples_features[np.newaxis, :, :], self.classes_features_mean[:, np.newaxis, :])
        classes_samples_logprob = np.sum(classes_samples_features_logprob, axis=2)
        samples_argmax = np.argmax(classes_samples_logprob, axis=0)
        return samples_argmax


class RidgeRegression:

    def __init__(self, ridge=0.):
        self._ridge = ridge
        self.XTX = None
        self.XTy = None
        self.n_train = None
        self.coeffs_matrix = None

    @property
    def ridge(self):
        return self._ridge

    @ridge.setter
    def ridge(self, value):
        if value < 0:
            raise ValueError("Ridge value must be a positive number.")
        if value == self._ridge:
            return
        self._ridge = value
        self.coeffs_matrix = None

    def fit(self, samples_features_train, samples_targets_train):
        self.XTX = samples_features_train.T @ samples_features_train
        self.XTy = samples_features_train.T @ samples_targets_train
        self.n_train = samples_features_train.shape[0]
        self.coeffs_matrix = None

    def predict(self, samples_features_test, ridge=None):
        if ridge is not None:
            self.ridge = ridge
        if self.coeffs_matrix is None:
            self.coeffs_matrix = (
                np.linalg.inv(self.XTX + (self.n_train * self.ridge * np.eye(self.XTX.shape[0])))
                @ self.XTy)
        return samples_features_test @ self.coeffs_matrix


class KalmanFilter:

    def __init__(self):
        self.A = None
        self.C = None
        self.W = None
        self.Q = None

    def fit(self, timesteps_features_train, timesteps_targets_train):
        n_timesteps = timesteps_features_train.shape[0]
        X_prev = timesteps_targets_train[:-1].T
        X_curr = timesteps_targets_train[1:].T
        Y_curr = timesteps_features_train[1:].T
        self.A = X_curr @ X_prev.T @ np.linalg.inv(X_prev @ X_prev.T)
        self.C = Y_curr @ X_curr.T @ np.linalg.inv(X_curr @ X_curr.T)
        X_curr_minus_A_X_prev = X_curr - self.A @ X_prev
        self.W = (1. / (n_timesteps - 1.)) * X_curr_minus_A_X_prev @ X_curr_minus_A_X_prev.T
        Y_curr_minus_C_X_curr = Y_curr - self.C @ X_curr
        self.Q = (1. / n_timesteps) * Y_curr_minus_C_X_curr @ Y_curr_minus_C_X_curr.T

    def predict(self, timesteps_features_test, features_init):
        predictions = []
        x = np.array(features_init).reshape((-1, 1))
        P = self.W
        for i, y in enumerate(timesteps_features_test):
            x_priori = self.A @ x
            P_priori = self.A @ P @ self.A.T + self.W
            K = P_priori @ self.C.T @ np.linalg.inv(self.C @ P_priori @ self.C.T + self.Q)
            x = x_priori + K @ (y.reshape((-1, 1)) - self.C @ x_priori)
            predictions.append(x.flatten())
            P = (np.eye(K.shape[0]) - K @ self.C) @ P_priori
        return np.array(predictions)


def main(contdata95_path, ecogclassifydata_path, firingrate_path, output_path):

    # =========================================================================
    # Data Loading
    # =========================================================================

    contdata_dict = loadmat(contdata95_path)
    np_contdata_x = contdata_dict["X"]
    np_contdata_y = contdata_dict["Y"]

    ecog_dict = loadmat(ecogclassifydata_path)
    np_ecog_powervals = ecog_dict["powervals"]
    np_ecog_true_classes = ecog_dict["group"].flatten()
    np_firing_rate = loadmat(firingrate_path)["firingrate"].swapaxes(0, 2)  # (class, sample, feature)

    # =========================================================================
    # Naive Bayes
    # =========================================================================

    # Split data into train and test
    n_classes, n_samples_per_class, n_features = np_firing_rate.shape
    n_samples_per_class_train = int(n_samples_per_class // 2)
    n_samples_per_class_test = n_samples_per_class - n_samples_per_class_train
    n_samples_train = n_classes * n_samples_per_class_train
    n_samples_test = n_classes * n_samples_per_class_test
    np_firing_rate_train = np_firing_rate[:, :n_samples_per_class_train, :]
    np_firing_rate_test = np_firing_rate[:, n_samples_per_class_train:, :].reshape((n_samples_test, n_features))
    np_true_classes_test = (
            np.ones(shape=(1, n_samples_per_class_test), dtype=int)
            * np.arange(n_classes, dtype=int).reshape((n_classes, 1))).reshape((n_samples_test, ))

    # Train classifier and perform predictions
    classifier = NaiveBayesClassifier()
    classifier.fit(np_firing_rate_train)
    np_pred_classes_test = classifier.predict(np_firing_rate_test)

    # Evaluate prediction accuracy
    naive_bayes_test_acc = np.mean(np_true_classes_test == np_pred_classes_test)
    print(f"Naive Bayes Accuracy (test data): {naive_bayes_test_acc*100:2.2f}%")

    # Generate and test fake data
    np_features_mean = np.mean(np_firing_rate, axis=(0, 1))
    np_firing_rate_fake = poisson.rvs(mu=np_features_mean.reshape((1, n_features)), size=(n_samples_test, n_features))
    np_pred_classes_fake = classifier.predict(np_firing_rate_fake)
    naive_bayes_fake_acc = np.mean(np_true_classes_test == np_pred_classes_fake)
    print(f"Naive Bayes Accuracy (fake data): {naive_bayes_fake_acc * 100:2.2f}%")

    # =========================================================================
    # Linear Discriminant Analysis
    # =========================================================================

    # Leave-one-out cross-validation with LDA
    n_samples, n_features = np_ecog_powervals.shape
    np_ecog_pred_classes = np.empty(shape=(n_samples, ), dtype=int)
    for test_sample_idx in range(n_samples):
        list_ecog_power_train = np.concatenate(
            [np_ecog_powervals[:test_sample_idx], np_ecog_powervals[test_sample_idx+1:]])
        list_ecog_class_train = np.concatenate(
            [np_ecog_true_classes[:test_sample_idx], np_ecog_true_classes[test_sample_idx+1:]])
        classifier = LinearDiscriminantAnalysis()
        classifier.fit(list_ecog_power_train, list_ecog_class_train)
        np_ecog_pred_classes[test_sample_idx] = (classifier.predict(np_ecog_powervals[test_sample_idx].reshape((1, -1))))

    # Evaluate prediction accuracy
    lda_test_acc = np.mean(np_ecog_true_classes == np_ecog_pred_classes)
    print(f"LDA Accuracy: {lda_test_acc * 100:2.2f}%")

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true=np_ecog_true_classes, y_pred=np_ecog_pred_classes, colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "lda_confusion.png"))
    plt.close()

    # =========================================================================
    # Linear Regression
    # =========================================================================

    # Preprocess dataset (split train/test first to ensure no rolling window overlap)
    window_size = 10
    n_timesteps, n_neurons = np_contdata_y.shape
    n_timesteps_train_raw = int(n_timesteps // 2)
    np_contdata_x_train = np_contdata_x[window_size - 1:n_timesteps_train_raw, :]
    np_contdata_x_test = np_contdata_x[n_timesteps_train_raw + window_size - 1:, :]
    np_contdata_y_train_raw = np_contdata_y[:n_timesteps_train_raw, :]
    np_contdata_y_test_raw = np_contdata_y[n_timesteps_train_raw:, :]
    np_contdata_y_train = np.lib.stride_tricks.sliding_window_view(
        np_contdata_y_train_raw, window_shape=window_size, axis=0).swapaxes(1, 2).reshape((-1, window_size * n_neurons))
    np_contdata_y_test = np.lib.stride_tricks.sliding_window_view(
        np_contdata_y_test_raw, window_shape=window_size, axis=0).swapaxes(1, 2).reshape((-1, window_size * n_neurons))
    np_contdata_y_train = np.pad(np_contdata_y_train, pad_width=((0, 0), (0, 1)), constant_values=1).astype(float)
    np_contdata_y_test = np.pad(np_contdata_y_test, pad_width=((0, 0), (0, 1)), constant_values=1).astype(float)
    n_samples_train, n_features = np_contdata_y_train.shape
    n_samples_test, n_features = np_contdata_y_test.shape

    # OLS Regression
    regressor = RidgeRegression(ridge=0.)
    regressor.fit(np_contdata_y_train, np_contdata_x_train)
    np_contdata_x_test_pred = regressor.predict(np_contdata_y_test)

    # Plot predicted vs actual movement
    start_idx = 0
    end_idx = 100
    fig, axes = plt.subplots(2, 1, sharex="all", sharey="none")
    for i, label in enumerate(("x_pos", "y_pos")):
        axes[i].plot(np_contdata_x_test[start_idx:end_idx, i], label="true")
        axes[i].plot(np_contdata_x_test_pred[start_idx:end_idx, i], label="pred")
        axes[i].legend()
        axes[i].set_ylabel(label)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ols_pred_vs_true.png"))
    plt.close()

    # Compute RMSE and correlation for each output target
    np_rmse = np.sqrt(np.mean(np.square(np_contdata_x_test_pred - np_contdata_x_test), axis=0))
    np_corr = [np.corrcoef(np_contdata_x_test_pred[:, i], np_contdata_x_test[:, i])[0, 1] for i in range(4)]
    for i, label in enumerate(("x_pos", "y_pos", "x_vel", "y_vel")):
        print(f"Linear regression RMSE for {label}: {np_rmse[i]:1.4f}")
    for i, label in enumerate(("x_pos", "y_pos", "x_vel", "y_vel")):
        print(f"Linear regression correlation for {label}: {np_corr[i]:1.4f}")

    # =========================================================================
    # Ridge Regression
    # =========================================================================

    # Optimize ridge for maximum total correlation (since correlation is scale invariant)
    # Optimization is done using 5-fold cross-validation on the training set
    # (doesn't make mathematical sense to optimize directly on the training set)
    n_folds = 5
    n_ridge_values = 15
    ridge_min = 0
    ridge_max = 0.1
    total_corrs = np.empty(shape=(n_folds, n_ridge_values))
    fold_bounds = np.round(np.linspace(0, n_samples_train, num=n_folds + 1)).astype(int)
    for fold_idx in range(n_folds):
        np_cvtrain_indexer = np.full(fill_value=True, shape=(n_samples_train, ), dtype=bool)
        np_cvval_indexer = np.full(fill_value=False, shape=(n_samples_train, ), dtype=bool)
        np_cvtrain_indexer[fold_bounds[fold_idx]:fold_bounds[fold_idx + 1]] = False
        np_cvval_indexer[fold_bounds[fold_idx]:fold_bounds[fold_idx + 1] - window_size + 1] = True
        np_contdata_y_cvtrain = np_contdata_y_train[np_cvtrain_indexer]
        np_contdata_x_cvtrain = np_contdata_x_train[np_cvtrain_indexer]
        np_contdata_y_cvval = np_contdata_y_train[np_cvval_indexer]
        np_contdata_x_cvval = np_contdata_x_train[np_cvval_indexer]
        regressor.fit(np_contdata_y_cvtrain, np_contdata_x_cvtrain)
        for ridge_idx, ridge in enumerate(np.linspace(ridge_min, ridge_max, num=n_ridge_values)):
            np_contdata_x_cvval_pred = regressor.predict(np_contdata_y_cvval, ridge=ridge)
            total_corrs[fold_idx, ridge_idx] = np.sum([
                np.corrcoef(np_contdata_x_cvval_pred[:, i], np_contdata_x_cvval[:, i])[0, 1]
                for i
                in range(4)])
    total_corrs = np.mean(total_corrs, axis=0)

    # Plot total correlation against ridge parameter
    fig, axes = plt.subplots(1, 1)
    axes.plot(np.linspace(ridge_min, ridge_max, num=n_ridge_values), total_corrs, marker="o")
    axes.set_xlabel("ridge")
    axes.set_ylabel("sum of correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ridge_param_crossval.png"))
    plt.close()

    # Identify optimal ridge parameter and retrain and re-evaluate on full training dataset
    ridge_best = np.linspace(ridge_min, ridge_max, num=n_ridge_values)[np.argmax(total_corrs)]
    print(f"Best ridge parameter found: {ridge_best:1.5f}")
    regressor.fit(np_contdata_y_train, np_contdata_x_train)
    np_contdata_x_test_pred = regressor.predict(np_contdata_y_test, ridge=ridge_best)
    np_rmse = np.sqrt(np.mean(np.square(np_contdata_x_test_pred - np_contdata_x_test), axis=0))
    np_corr = [np.corrcoef(np_contdata_x_test_pred[:, i], np_contdata_x_test[:, i])[0, 1] for i in range(4)]
    for i, label in enumerate(("x_pos", "y_pos", "x_vel", "y_vel")):
        print(f"Ridge regression RMSE for {label}: {np_rmse[i]:1.4f}")
    for i, label in enumerate(("x_pos", "y_pos", "x_vel", "y_vel")):
        print(f"Ridge regression correlation for {label}: {np_corr[i]:1.4f}")
    ridge_coeff_matrix_values = regressor.coeffs_matrix.flatten()

    # =========================================================================
    # LASSO Regression
    # =========================================================================

    # Fit and evaluate Lasso regression
    regressor = Lasso(alpha=ridge_best)
    regressor.fit(np_contdata_y_train, np_contdata_x_train)
    np_contdata_x_test_pred = regressor.predict(np_contdata_y_test)
    np_rmse = np.sqrt(np.mean(np.square(np_contdata_x_test_pred - np_contdata_x_test), axis=0))
    np_corr = [np.corrcoef(np_contdata_x_test_pred[:, i], np_contdata_x_test[:, i])[0, 1] for i in range(4)]
    for i, label in enumerate(("x_pos", "y_pos", "x_vel", "y_vel")):
        print(f"Lasso regression RMSE for {label}: {np_rmse[i]:1.4f}")
    for i, label in enumerate(("x_pos", "y_pos", "x_vel", "y_vel")):
        print(f"Lasso regression correlation for {label}: {np_corr[i]:1.4f}")

    # Inspect parameter vector value distribution
    lasso_coeff_matrix_values = regressor.coef_.flatten()
    fig, axes = plt.subplots(2, 1, sharex="all", sharey="all")
    for i, (label, values) in enumerate(zip(
            ("ridge", "lasso"),
            (ridge_coeff_matrix_values, lasso_coeff_matrix_values))):
        axes[i].set_yscale("log")
        axes[i].hist(values, range=(-5, 5), bins=100)
        axes[i].set_title(f"weight histogram for {label}")
    plt.savefig(os.path.join(output_path, "weight_histograms.png"))
    plt.close()

    # =========================================================================
    # Kalman Filter
    # =========================================================================

    kalman = KalmanFilter()
    kalman.fit(np_contdata_y_train, np_contdata_x_train)
    np_contdata_x_test_pred = kalman.predict(np_contdata_y_test, np_contdata_x_test[0])
    np_rmse = np.sqrt(np.mean(np.square(np_contdata_x_test_pred - np_contdata_x_test), axis=0))
    np_corr = [np.corrcoef(np_contdata_x_test_pred[:, i], np_contdata_x_test[:, i])[0, 1] for i in range(4)]
    for i, label in enumerate(("x_pos", "y_pos", "x_vel", "y_vel")):
        print(f"Kalman filter RMSE for {label}: {np_rmse[i]:1.4f}")
    for i, label in enumerate(("x_pos", "y_pos", "x_vel", "y_vel")):
        print(f"Kalman filter correlation for {label}: {np_corr[i]:1.4f}")





if __name__ == "__main__":

    _contdata95_path = r"D:\Documents\Academics\BME517\bme_lab_7_8_9\data\contdata95.mat"
    _ecogclassifydata_path = r"D:\Documents\Academics\BME517\bme_lab_7_8_9\data\ecogclassifydata.mat"
    _firingrate_path = r"D:\Documents\Academics\BME517\bme_lab_7_8_9\data\firingrate.mat"
    _output_path = r"D:\Documents\Academics\BME517\bme_lab_7_8_9_report"

    main(_contdata95_path, _ecogclassifydata_path, _firingrate_path, _output_path)

