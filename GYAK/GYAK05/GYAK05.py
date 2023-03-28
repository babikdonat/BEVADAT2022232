import numpy as np
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import seaborn as sns


class KNNClassifier:
    
    def __init__(self, k:int,test_split_ratio:float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio


    @property
    def k_neighbors(self):
        return self.k

    def train_set_split(self,features:np.ndarray,labels:np.ndarray)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        test_size = int(len(features)*self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size,'Size mismatch!'

        x_train,y_train = features[:train_size,:],labels[:train_size]
        x_test,y_test = features[train_size:,:],labels[train_size:]
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def euclidean(self,element_of_x:np.ndarray)->np.ndarray:
        return np.sqrt(np.sum((self.x_train - element_of_x)**2,axis=1))
    
    def predict(self,x_test:np.ndarray)->np.ndarray:
        labels_pred = []
        for x_test_element in x_test:
              distances = self.euclidean(self.x_train,x_test_element)
              distances = np.array(sorted(zip(distances,self.y_train)))

              label_pred = mode(distances[:self.k,1],keepdims=False).mode
              labels_pred.append(label_pred)

        self.y_preds = np.array(labels_pred)

    def accuracy(self)->float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100
    
    def plot_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test,self.y_preds)
        sns.heatmap(confusion_matrix,annot=True)

    @staticmethod
    def load_csv(csv_path:str)->Tuple[np.ndarray,np.ndarray]:
        np.random.seed(42)
        dataset = np.genfromtxt(csv_path,delimiter=',')
        np.random.shuffle(dataset)
        x,y = dataset[:,:-1],dataset[:,-1]
        return x,y
    

        
              