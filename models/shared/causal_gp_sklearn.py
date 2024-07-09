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



class CausalGPSklearn():
    def __init__(self, causal_var_info, kernel):
        super(CausalGPSklearn, self).__init__()
        self.causal_var_info = causal_var_info
        self.gp_list = self.get_gp_model_vec(causal_var_info, kernel)

    def get_gp_model_vec(self, var_info, kernel):
        gp = []

        for idx, val in enumerate(var_info.values()):
            if val.startswith('categ_'):
                gp.append(GaussianProcessClassifier(kernel=kernel))
            elif val.startswith('continuous_'):
                gp.append(GaussianProcessRegressor(kernel=kernel))

        return gp

    def forward(self, x):
        probas = []
        values_std = []
        values = []

        for gp in self.gp_list:
            if isinstance(gp, GaussianProcessClassifier):
                probas.append(gp.predict_proba(x))
            #values_std.append(gp.predict(x, return_std=True))
            values.append(gp.predict(x))

        return values, probas#, values_std

    def train(self, x, y):
        for index, gp in enumerate(self.gp_list):
            start_time = datetime.now()
            gp.fit(x, y[:, index])
            end_time = datetime.now()
            print(f'{index} gp finished training (Duration: {end_time - start_time})')

            if isinstance(gp, GaussianProcessClassifier):
                predictions = gp.predict(x)
                accuracy = accuracy_score(y[:, index], predictions)
                print(f'{index} gp accuracy train: {accuracy}')
            else:
                predictions = gp.predict(x)
                mse = mean_squared_error(y[:, index], predictions)
                print(f'{index} gp MSE train: {mse}')

    def save_gp_model(self, file_path):
        for i, gp in enumerate(self.gp_list):
            # TODO ordering might be an issue
            model = list(self.causal_var_info[i])
            joblib.dump(gp, os.path.join(file_path, model))
            print(f'Model {model} saved to {file_path}')


    def _get_loss(self, inps, target):
        values, probas = self.forward(inps)
        total_mse = 0
        total_log_loss = 0
        num_categorical = 0
        num_continuous = 0

        for idx, val in enumerate(self.causal_var_info.values()):
            if val.startswith('categ_'):
                total_log_loss += log_loss(np.array(target[:, idx]), np.array(values[idx]))
                num_categorical += 1
            elif val.startswith('continuous_'):
                total_mse += mean_squared_error(target[:, idx], values[idx])
                num_continuous += 1

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
