import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
import joblib
from datetime import datetime
import json



class CausalGPSklearn():
    def __init__(self, causal_var_info, kernel, result_path):
        super(CausalGPSklearn, self).__init__()
        self.causal_var_info = causal_var_info
        self.gp_dict = self.get_gp_model_vec(causal_var_info, kernel)
        self.result_path = result_path
        self.train_metrics = []
        self.test_metrics = []

    def get_gp_model_vec(self, var_info, kernel):
        gp_dict = {}

        for key, val in var_info.items():
            if val.startswith('categ_'):
                gp_dict[key] = GaussianProcessClassifier(kernel=kernel)
            elif val.startswith('continuous_'):
                gp_dict[key] = GaussianProcessRegressor(kernel=kernel)

        return gp_dict

    def forward(self, x):
        probas = {}
        values = {}

        for causal, gp in self.gp_dict.items():
            if isinstance(gp, GaussianProcessClassifier):
                probas[causal] = gp.predict_proba(x)
            else:
                probas[causal] = gp.predict(x, return_std=True)
            values[causal] = gp.predict(x)

        return values, probas

    def train(self, x, y):
        for index, (causal, gp) in enumerate(self.gp_dict.items()):
            start_time = datetime.now()
            gp.fit(x, y[:, index])
            end_time = datetime.now()
            print(f'{causal} gp finished training (Duration: {end_time - start_time})')
            train_metric = {'causal': causal, 'split': 'train'}
            if isinstance(gp, GaussianProcessClassifier):
                predictions = gp.predict(x)

                logloss = log_loss(y[:, index], predictions)
                print(f'{causal} gp log loss train: {logloss}')
                train_metric['loss'] = logloss

                accuracy = accuracy_score(y[:, index], predictions)
                print(f'{causal} gp accuracy train: {accuracy}')
                train_metric['accuracy'] = accuracy
            else:
                predictions = gp.predict(x)
                mse = mean_squared_error(y[:, index], predictions)
                print(f'{causal} gp MSE train: {mse}')
                train_metric['loss'] = mse
            self.train_metrics.append(train_metric)
        self._save_metrics_to_file(self.train_metrics, file_name=self.result_path)

    def _save_metrics_to_file(self, metrics, file_name):
        if os.path.exists(file_name):
            print("file exists")
            with open(file_name, 'r') as f:
                existing_metrics = json.load(f)
            existing_metrics.extend(metrics)
            print(existing_metrics)
            with open(file_name, 'w') as f:
                json.dump(existing_metrics, f, indent=4)
        else:
            with open(file_name, 'w') as f:
                json.dump(metrics, f, indent=4)

    def save_gp_model(self, file_path):
        for key, gp in self.gp_dict.items():
            joblib.dump(gp, os.path.join(file_path, key))
            print(f'Model {key} saved to {file_path}')


    def _get_loss(self, inps, target):
        values, probas = self.forward(inps)
        total_mse = 0
        total_log_loss = 0
        num_categorical = 0
        num_continuous = 0

        for idx, (causal, values_per_causal) in enumerate(values.items()):
            var_type = self.causal_var_info[causal]
            test_metric = {'causal': causal, 'split': 'test'}
            if var_type.startswith('categ_'):
                cur_loss = log_loss(np.array(target[:, idx]), np.array(values_per_causal))
                total_log_loss += cur_loss
                num_categorical += 1
                print(f'{causal} gp log loss test: {cur_loss}')
                test_metric['loss'] = cur_loss

                cur_acc = accuracy_score(np.array(target[:, idx]), np.array(values_per_causal))
                print(f'{causal} gp accuracy test: {cur_acc}')
                test_metric['accuracy'] = cur_acc
            elif var_type.startswith('continuous_'):
                cur_loss = mean_squared_error(target[:, idx], values_per_causal)
                total_mse += cur_loss
                num_continuous += 1
                print(f'{causal} gp MSE train test: {cur_loss}')
                test_metric['loss'] = cur_loss
            self.test_metrics.append(test_metric)
        self._save_metrics_to_file(self.test_metrics, file_name=self.result_path)
        avg_mse = total_mse / num_continuous if num_continuous > 0 else 0
        avg_log_loss = total_log_loss / num_categorical if num_categorical > 0 else 0
        combined_loss = avg_mse + avg_log_loss
        return combined_loss


    def uncertainty_contin(self, X_train, y_train, X_test, y_mean, y_std):
        plt.figure(figsize=(10, 5))

        # Plot the training data
        plt.scatter(X_train, y_train, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
        plt.plot(X_train, y_train, 'r.', markersize=10, label='Training Data')

        # Plot the predictive mean
        plt.plot(X_test, y_mean, 'k', lw=2, zorder=9, label='Predictive Mean')

        # Plot the 95% confidence interval (mean ± 1.96 * std)
        plt.fill_between(X_test.ravel(), y_mean - 1.96 * y_std, y_mean + 1.96 * y_std, alpha=0.2, color='k',
                         label='95% Confidence Interval')

        plt.title('Gaussian Process Regression')
        plt.xlabel('Input Dimension')
        plt.ylabel('Output Dimension')
        plt.legend()
        plt.show()

    def uncertainty_contin_multidim(self, y_mean, y_std, X1, X2, X_train):
        y_mean = y_mean.reshape(X1.shape)
        y_std = y_std.reshape(X1.shape)

        # Plot the predictive mean
        plt.figure(figsize=(10, 5))
        plt.contourf(X1, X2, y_mean, levels=20, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Predictive Mean')

        # Plot the 95% confidence interval (mean ± 1.96 * std)
        plt.contour(X1, X2, y_mean - 1.96 * y_std, levels=[0], colors='k', linestyles='--')
        plt.contour(X1, X2, y_mean + 1.96 * y_std, levels=[0], colors='k', linestyles='--')

        # Plot the training data
        plt.scatter(X_train[:, 0], X_train[:, 1], c='r', edgecolor='k', s=100, label='Training Data')

        plt.title('Gaussian Process Regression (2D)')
        plt.xlabel('Input Dimension 1')
        plt.ylabel('Input Dimension 2')
        plt.legend()
        plt.show()

    def uncertainty_categ(self, X1, X2, y_prob, X_train, y_train):
        plt.figure(figsize=(10, 8))

        # Plot the probability of the positive class
        contour = plt.contourf(X1, X2, y_prob, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(contour, label='Probability of Positive Class')

        # Plot the decision boundary (probability = 0.5)
        plt.contour(X1, X2, y_prob, levels=[0.5], colors='red', linestyles='--')

        # Plot the training data
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap='viridis', s=100, marker='o',
                    label='Training Data')

        plt.title('Gaussian Process Classification')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

    def uncertainty_entropy(self, y_prob, X1, X2, X_train, y_train):
        from scipy.stats import entropy

        # Compute the entropy of the predicted probabilities
        entropy_values = entropy([y_prob, 1 - y_prob], base=2)

        # Reshape the entropy values to the grid shape
        entropy_values = entropy_values.reshape(X1.shape)

        plt.figure(figsize=(10, 8))

        # Plot the entropy of the probabilities
        contour = plt.contourf(X1, X2, entropy_values, levels=20, cmap='coolwarm', alpha=0.8)
        plt.colorbar(contour, label='Entropy (Uncertainty)')

        # Plot the training data
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', cmap='coolwarm', s=100, marker='o',
                    label='Training Data')

        plt.title('Uncertainty Visualization for Gaussian Process Classification')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
