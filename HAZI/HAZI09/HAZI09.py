# %%
# imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

# %%
# Készíts egy függvényt ami betölti a digits datasetet 
# NOTE: használd az sklearn load_digits-et
# Függvény neve: load_digits()
# Függvény visszatérési értéke: a load_digits visszatérési értéke
from sklearn.datasets import load_digits as sk_load_digits

def load_digits():
    return sk_load_digits()

# %%
# Vizsgáld meg a betöltött adatszetet (milyen elemek vannak benne stb.)

dataset = load_digits()
print(dataset.feature_names)
print(dataset.target_names)
print(dataset.target)

# %%
# Vizsgáld meg a data paraméterét a digits dataset-nek (tartalom,shape...)
print(dataset.data.shape)
print(dataset.data)

# %%
# Készíts egy függvényt ami létrehoz egy KMeans model-t 10 db cluster-el
# NOTE: használd az sklearn Kmeans model-jét (random_state legyen 0)
# Miután megvan a model predict-elj vele 
# NOTE: használd a fit_predict-et
# Függvény neve: predict(n_clusters:int,random_state:int,digits)
# Függvény visszatérési értéke: (model:sklearn.cluster.KMeans,clusters:np.ndarray)
def predict(n_clusters:int,random_state:int,dataset):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = model.fit_predict(dataset.data, dataset.target)
    return model, clusters


model, clusters = predict(10, 0, dataset)

# %%
# Vizsgáld meg a shape-jét a kapott model cluster_centers_ paraméterének.
print(model.cluster_centers_.shape)

# %%
# Készíts egy plotot ami a cluster középpontokat megjeleníti
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1])

# %%
# Készíts egy függvényt ami visszaadja a predictált cluster osztályokat
# NOTE: amit a predict-ből visszakaptunk "clusters" azok lesznek a predictált cluster osztályok
# HELP: amit a model predictált cluster osztályok még nem a labelek, hanem csak random cluster osztályok, 
#       Hogy label legyen belőlük:
#       1. készíts egy result array-t ami ugyan annyi elemű mint a predictált cluster array
#       2. menj végig mindegyik cluster osztályon (0,1....9)
#       3. készíts egy maszkot ami az adott cluster osztályba tartozó elemeket adja vissza
#       4. a digits.target-jét indexeld meg ezzel a maszkkal
#       5. számold ki ennel a subarray-nek a móduszát
#       6. a result array-ben tedd egyenlővé a módusszal azokat az indexeket ahol a maszk True 
#       Erre azért van szükség mert semmi nem biztosítja nekünk azt, hogy a "0" cluster a "0" label lesz, lehet, hogy az "5" label lenne az.

# Függvény neve: get_labels(clusters:np.ndarray, digits)
# Függvény visszatérési értéke: labels:np.ndarray
def get_labels(clusters:np.ndarray, dataset)-> np.ndarray:
    labels = np.empty(shape=clusters.shape)

    for i in dataset.target_names:
        mask = clusters == i
        labels[mask] = mode(dataset.target[mask], keepdims=False).mode
    
    return labels

        
labels = get_labels(clusters, dataset)

# %%
# Készíts egy függvényt ami kiszámolja a model accuracy-jét
# Függvény neve: calc_accuracy(target_labels:np.ndarray,predicted_labels:np.ndarray)
# Függvény visszatérési értéke: accuracy:float
# NOTE: Kerekítsd 2 tizedes jegyre az accuracy-t
def calc_accuracy(target_labels:np.ndarray,predicted_labels:np.ndarray):
    return np.round(accuracy_score(target_labels, predicted_labels), 2)

calc_accuracy(dataset.target, labels)

# %%
# Készíts egy confusion mátrixot és plot-old seaborn segítségével
conf_matrix = confusion_matrix(dataset.target, labels)

sns.heatmap(conf_matrix)


