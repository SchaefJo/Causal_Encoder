import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
import joblib
from datetime import datetime
import json



class CausalUncertaintySklearn():
    def __init__(self, model_name, causal_var_info, result_path, kernel=None):
        super(CausalUncertaintySklearn, self).__init__()
        self.causal_var_info = causal_var_info
        self.gp_dict = self.get_gp_model_vec(causal_var_info, kernel)
        self.result_path = result_path
        self.train_metrics = []
        self.test_metrics = []
        self.model_name = model_name

    def get_gp_model_vec(self, var_info, kernel):
        gp_dict = {}

        for key, val in var_info.items():
            if val.startswith('categ_'):
                if self.model_name == 'gp':
                    gp_dict[key] = GaussianProcessClassifier(kernel=kernel)
                elif self.model_name == 'rf':
                    gp_dict[key] = RandomForestClassifier()
            elif val.startswith('continuous_'):
                if self.model_name == 'gp':
                    gp_dict[key] = GaussianProcessRegressor(kernel=kernel)
                elif self.model_name == 'rf':
                    gp_dict[key] = RandomForestRegressor()

        return gp_dict

    def calculate_entropy(self, probas):
        epsilon = 1e-10
        probas = np.clip(probas, epsilon, 1 - epsilon)
        entropy = -np.sum(probas * np.log(probas), axis=1)
        return entropy

    def forward(self, x):
        probas = {}
        values = {}

        for causal, gp in self.gp_dict.items():
            if (isinstance(gp, GaussianProcessClassifier)) or (isinstance(gp, RandomForestClassifier)):
                probas[causal] = self.calculate_entropy(gp.predict_proba(x))
            elif isinstance(gp, GaussianProcessRegressor):
                _, probas[causal] = gp.predict(x, return_std=True)
            elif isinstance(gp, RandomForestRegressor):
                all_tree_preds = np.array([tree.predict(x) for tree in gp.estimators_])
                probas[causal] = np.std(all_tree_preds, axis=0)
            values[causal] = gp.predict(x)

        return values, probas

    def train(self, x, y, verbose=True, save=True):
        for index, (causal, gp) in enumerate(self.gp_dict.items()):
            start_time = datetime.now()
            gp.fit(x, y[:, index])
            end_time = datetime.now()
            print(f'{causal} gp finished training (Duration: {end_time - start_time})')
            train_metric = {'causal': causal, 'split': 'train'}
            if verbose:
                if isinstance(gp, GaussianProcessClassifier) or isinstance(gp, RandomForestClassifier):
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
        if save:
            self._save_metrics_to_file(self.train_metrics, file_name=self.result_path)

    def _save_metrics_to_file(self, metrics, file_name):
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                existing_metrics = json.load(f)
            existing_metrics.extend(metrics)
            with open(file_name, 'w') as f:
                json.dump(existing_metrics, f, indent=4)
        else:
            with open(file_name, 'w') as f:
                json.dump(metrics, f, indent=4)

    def save_gp_model(self, file_path):
        for key, gp in self.gp_dict.items():
            joblib.dump(gp, os.path.join(file_path, key))
            print(f'Model {key} saved to {file_path}')


    def _get_loss(self, inps, target, save=True, al_iter=None):
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
                #print(f'{causal} gp log loss test: {cur_loss}')
                test_metric['loss'] = cur_loss

                cur_acc = accuracy_score(np.array(target[:, idx]), np.array(values_per_causal))
                #print(f'{causal} gp accuracy test: {cur_acc}')
                test_metric['accuracy'] = cur_acc
            elif var_type.startswith('continuous_'):
                cur_loss = mean_squared_error(target[:, idx], values_per_causal)
                total_mse += cur_loss
                num_continuous += 1
                #print(f'{causal} gp MSE train test: {cur_loss}')
                test_metric['loss'] = cur_loss
            if al_iter is not None:
                test_metric["al_iter"] = al_iter
            self.test_metrics.append(test_metric)
        if save:
            self._save_metrics_to_file(self.test_metrics, file_name=self.result_path)
        avg_mse = total_mse / num_continuous if num_continuous > 0 else 0
        avg_log_loss = total_log_loss / num_categorical if num_categorical > 0 else 0
        combined_loss = avg_mse + avg_log_loss
        print(f'Combined Loss: {combined_loss}')
        return combined_loss

