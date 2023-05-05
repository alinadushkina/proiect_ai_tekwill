import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer( )

df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

d = df
X = d.drop(['target'], axis=1)
y = d['target']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X=pca.fit_transform(X)
print(pca.explained_variance_ratio_)

from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)

from sklearn.model_selection import train_test_split
from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(X, y, train_size=0.8, random_state=algorithm_globals.random_seed)

from sklearn.svm import SVC
classical_kernel = ['rbf', 'linear', 'poly', 'sigmoid']

start = time.time()
for kernel in classical_kernels:
    classical_svc = SVC(kernel=kernel)
    classical_svc.fit(train_features, train_labels)
    classical_score = classical_svc.score(test_features, test_labels)

    print(f'{kernel} kernel classification test score: {classical_score:.2f}')

elapsed = time.time() - start
print(f'Training and testing time: {elapsed} seconds')

from qiskit.circuit.library import ZZFeatureMap

num_features = train_features.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=3, entaglement="linear", insert_barriers=True)
feature_map.decompose().draw(output='mpl')

from qiskit import Aer, execute
backend = Aer.get_backend('qasm_simulator')

from qiskit import IBMQ
IBMQ.save account('<your TOKEN>')

IBMQ.load_account()
provider = IBMQ.get_backend('ibmq_manila')

from qiskit_machine_learning_kernels import QuantumKernel
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)

quantum_svc = SVC(kernel=quantum_kernel.evaluate)

quantum_svc.fit(train_features, train_labels)
quantum_score = quantum_svc.score(test_features, test_labels)

print(f'Callable quantum kernel classification test score: {quantum_score}')

matrix_train = quantum_kernel.evaluate(x_vec=train_features)
matrix_test = quantum_kernel.evaluate(x_vec=test_features, y_vec=train_features)

quantum_svc = SVC(kernel='precomputed')
quantum_svc.fit(matrix_train, train_labels)
quantum_score = quantum_svc.score(matrix_test, test_labels)

print(f'Quantum kernel classification test score: {quantum_score}')