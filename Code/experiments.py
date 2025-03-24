import ConfigSpace
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.model_selection
import sklearn.svm
import typing
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

from assignment import SequentialModelBasedOptimization

np.random.seed(0)

# variable to run code either with SVM or Adaboost
svm_or_ada = 2

# Datasets ids: 1478, 1501, 16
id = 16
print("Dataset ", id)
data = sklearn.datasets.fetch_openml(data_id=id)
df = pd.DataFrame(data=data.data, columns=data.feature_names)
print(df.shape)
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
    data.data, data.target, test_size=0.33, random_state=1)

def optimizee(gamma, C):
    clf = sklearn.svm.SVC()
    clf.set_params(kernel='rbf', gamma=gamma, C=C)
    clf.fit(X_train, y_train)
    return sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid))

def optimize_adaboost(theta_new0, theta_new1):
    clf = sklearn.svm.SVC()  # generates classifier
    clf.set_params(kernel='rbf', gamma=theta_new0, C=theta_new1)
    classifier = AdaBoostClassifier(n_estimators=50, estimator=clf,learning_rate=1, algorithm='SAMME')
    classifier.fit(X_train, y_train)
    return sklearn.metrics.accuracy_score(y_valid, classifier.predict(X_valid))

def sample_configurations(
        n_configurations,
        config_space: ConfigSpace.ConfigurationSpace
):
    return [(configuration['gamma'],
             configuration['C'])
            for configuration in config_space.sample_configuration(n_configurations)]


def sample_initial_configurations(
        n: int,
        config_space: ConfigSpace.ConfigurationSpace
) -> typing.List[typing.Tuple[np.array, float]]:
    configs = sample_configurations(n, config_space=config_space)
    return [((gamma, C), optimizee(gamma, C)) for gamma, C in configs]


def plot_performance(performances):
    evaluations = list(range(1, 17))
    plt.figure(figsize=(8, 6))
    plt.ylim(0, 1)
    plt.plot(evaluations, performances, marker='o', linestyle='-')
    if svm_or_ada == 1:
        plt.title('SMBO optimizing SVM Convergence Plot')
    else:
        plt.title('SMBO optimizing Adaboost Convergence Plot')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.show()


config_space = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', 1)
C = ConfigSpace.UniformFloatHyperparameter(
    name='C', lower=0.03125, upper=32768, log=True, default_value=1.0)
gamma = ConfigSpace.UniformFloatHyperparameter(
    name='gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
config_space.add_hyperparameters([C, gamma])

smbo = SequentialModelBasedOptimization()
smbo.initialize(sample_initial_configurations(10, config_space))

performances = []
configurations = []

for idx in range(16):
    if svm_or_ada == 1:
        print('SVM iteration %d/16' % (idx+1))
    else:
        print('Adaboost iteration %d/16' % (idx+1))
    smbo.fit_model()
    theta_new = smbo.select_configuration(sample_configurations(64, config_space))
    if svm_or_ada == 1:
        performance = optimizee(theta_new[0], theta_new[1])
    else:
        performance = optimize_adaboost(theta_new[0], theta_new[1])
    smbo.update_runs((theta_new, performance))

    performances.append(performance)
    configurations.append(theta_new)
    if svm_or_ada == 1:
        print("SVM configuration: ", theta_new, ", performance = ", performance)
    else:
        print("Adaboost configuration: ", theta_new, ", performance = ", performance)


# Best performance
optimal_performance = max(performances)
max_index = np.argmax(performances)
optimal_configuration = configurations[max_index]
if svm_or_ada == 1:
    print("\nBest SMBO optimizing SVM performance = ", optimal_performance, ", with configuration: ", optimal_configuration)
else:
    print("\nBest SMBO optimizing Adaboost performance = ", optimal_performance, ", with configuration: ", optimal_configuration)
smbo_end = time.time()


def param_grid():
    configurations = sample_configurations(16, config_space)
    param_grid = {"C": [], "gamma": []}
    for gamma, C in configurations:
        param_grid["C"].append(C)
        param_grid["gamma"].append(gamma)
    return param_grid

# GRID SEARCH
def grid_search(param_grid):
    svm_classifier = sklearn.svm.SVC(kernel='rbf')
    grid = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, n_jobs=-1) # cv = 5
    grid.fit(X_train, y_train)
    print("Grid search best performance = ", grid.best_score_, ", with configuration ",
          grid.best_params_)

grid_search(param_grid())

# RANDOM SEARCH
def random_search(param_grid):
    svm_classifier = sklearn.svm.SVC(kernel='rbf')
    random_search = RandomizedSearchCV(svm_classifier, param_distributions=param_grid, n_iter=50, n_jobs=-1) #cv = 5
    random_search.fit(X_train, y_train)
    print("Random search best performance = ", random_search.best_score_, ", with configuration ",
          random_search.best_params_)

random_search(param_grid())




