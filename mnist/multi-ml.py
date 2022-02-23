import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import time
from tabulate import tabulate

class MultiML():
    def __init__(self, csv_location, first_feat_col_name, last_feat_col_name, label_col_name, nb_type):
        # the label must be either the first column or last column
        # first_feat_col_name and last_feat_col_name are the names of the first and last feature columns
        # label_col_name is the name of the column which contains the labels
 
        self.csv_location = csv_location
        self.first_feat_col_name = first_feat_col_name
        self.last_feat_col_name = last_feat_col_name
        self.label_col_name = label_col_name
        self.nb_type = nb_type

        self.all_classifiers_trained = False
        self.accuracy_calculated = False

    def import_data(self):
        # IMPORT DATA FROM FILE
        df = pd.read_csv(self.csv_location)

        # SPLIT DATA INTO TRAIN AND TEST
        train = df.sample(frac=0.7)
        test = df.drop(train.index)

        # SPLIT DATA INTO FEATURES AND LABELS
        self.X_train = train.loc[:, self.first_feat_col_name:self.last_feat_col_name]
        self.y_train = train.loc[:, self.label_col_name]

        self.X_test = test.loc[:, self.first_feat_col_name:self.last_feat_col_name]
        self.y_test = test.loc[:, self.label_col_name]

    def naive_bayes(self):
        if self.nb_type == "categorical":
            return {"name":"Naive Bayes (Categorical)", "classifier":CategoricalNB()}
        
        elif self.nb_type == "gaussian":
            return {"name":"Naive Bayes (Gaussian)", "classifier":GaussianNB()}
        
        elif self.nb_type == "multinomial":
            return {"name":"Naive Bayes (Multinomial)", "classifier":MultinomialNB()}
        
        elif self.nb_type == "complement":
            return {"name":"Naive Bayes (Complement)", "classifier":ComplementNB()}
        
        elif self.nb_type == "bernoulli":
            return {"name":"Naive Bayes (Bernoulli)", "classifier":BernoulliNB()}

    def compile_classifiers(self):
        self.classifiers = []

        print("Compiling classifiers...")
        
        self.classifiers.append({"name":"k-Nearest Neighbor (n=3)", "classifier":KNeighborsClassifier(n_neighbors=3)})
        self.classifiers.append(self.naive_bayes())
        self.classifiers.append({"name":"Logistic Regression", "classifier":LogisticRegression(random_state=0)})
        self.classifiers.append({"name":"Perceptron", "classifier":Perceptron(tol=1e-3, random_state=0)})
        self.classifiers.append({"name":"Multi-Layer Perceptron (64-64)", "classifier":MLPClassifier(random_state=1, max_iter=100, hidden_layer_sizes=(64, 64))})
        self.classifiers.append({"name":"Decision Tree", "classifier":DecisionTreeRegressor()})
        
        print("")

    def train_classifiers(self):
        for classifier in self.classifiers:
            print(f'Training {classifier["name"]} classifier...')
            
            start = time.time()
            classifier["classifier"].fit(self.X_train, self.y_train)
            classifier["train time"] = time.time() - start

        print("")

        self.all_classifiers_trained = True

    def test_accuracy(self):
        if self.all_classifiers_trained:
            for classifer in self.classifiers:
                print(f'Testing {classifer["name"]} classifier...')
                
                start = time.time()
                y_pred = classifer["classifier"].predict(self.X_test)
                classifer["test time"] = ((time.time() - start) / len(self.X_test))*1000000
                
                classifer["accuracy"] = accuracy_score(self.y_test, y_pred)

            print("")

            self.accuracy_calculated = True
        else:
            print("Some or all of the classifiers have not been trained.")

    def print_accuracy(self):
        tabulate_lst = []
        if self.accuracy_calculated:
            for classifier in self.classifiers:
                trt = str(round(classifier["train time"], 2))
                tet = str(int(classifier["test time"]))
                acc = str(round(classifier["accuracy"]*100, 1))
                tabulate_lst.append([classifier["name"], trt, tet, acc])
            
            print(tabulate(tabulate_lst, headers=['Classifier', 'Training Time (s)', 'Testing Time (μs)', 'Accuracy (%)']))

        else:
            print("Accuracy has not been calculated yet.")
    
    def run(self):
        self.import_data()
        self.compile_classifiers()
        self.train_classifiers()
        self.test_accuracy()
        self.print_accuracy()


if __name__ == "__main__":
    # clfs = MultiML("mnist-train-data.csv", "pixel1", "pixel778", "label", "bernoulli")
    # clfs.run()

    clfs = MultiML("phone-price/train.csv", "battery_power", "wifi", "price_range", "multinomial")
    clfs.run()

    
    
