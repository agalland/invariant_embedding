from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlxtend.data import loadlocal_mnist
from sklearn.svm import SVC

from utils import *

import time


graphs_name = ['PROTEINS']
data_pwd = "data/graph_classification/"

# params = {'emb_space': [2, 3, 4, 5, 10],
#           'emb_spec': [2, 3, 5, 8, 10, 20],
#           'max_num_nodes': [2, 3, 5, 8, 10, 20],
#           "t0": [11, 21, 31],
#           "t1": [11, 21, 31]}
params = {'emb_space': [3, 5], 'emb_spec': [20, 50], "t0": [21, 31], "t1": [21, 31]}

emb_space = None
emb_spec = None
max_num_nodes = None
t0 = None
t1 = None

if graphs_name[0] != "mnist":
    datasets = {}
    for gn in graphs_name:
        datasets[gn] = GraphDataset(folder_path=data_pwd + gn + '/' + gn, with_node_features=True)

if graphs_name[0] == "mnist":
    datasets = {}
    n_graphs = 100
    dataset = GraphDataset(folder_path=data_pwd,
                           with_node_features=True,
                           n_graphs=n_graphs)
    X = dataset.graphs_
    y = dataset.labels_
    datasets["mnist"] = dataset

for name, dataset in datasets.items():
    print(name)
    for emb_space in params["emb_space"]:
        for emb_spec_id, emb_spec in enumerate(params["emb_spec"]):
            for t0 in params["t0"]:
                for t1 in params["t1"]:

                    skf_0 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
                    X = np.zeros((dataset.n_graphs_, 2))
                    y = dataset.labels_
                    split_0 = skf_0.split(X, y)

                    list_split_0 = list(split_0)
                    accuraciesVal = []
                    for train_indices_0, val in list_split_0:
                        print("--------------------")
                        print("new fold")
                        print("--------------------")
                        max_accuracy_test = 0.
                        accuracy_cross_val = 0.

                        t = time.time()

                        # for k in params.keys():
                        #    if k == 'emb_space':
                        #        emb_space = params[k]
                        #    if k == 'emb_spec':
                        #        emb_spec = params[k]
                        #    if k == 't0':
                        #        t0 = params[k]
                        #    if k == 't1':
                        #        t1 = params[k]
                        #    for param_value in params[k]:

                        list_graphs = dataset_embedding(dataset,
                                                        emb_space,
                                                        emb_spec,
                                                        dataset.max_num_nodes,
                                                        t0,
                                                        t1)

                        n_samples = len(list_graphs)
                        Features = np.zeros((n_samples, list_graphs[0][0].shape[0]))
                        labels = np.zeros(n_samples)

                        for i, (x, l) in enumerate(list_graphs):
                            Features[i, :] = x
                            labels[i] = l

                        # print(emb_space, emb_spec, max_num_nodes, t0, t1)

                        #skf_1 = StratifiedKFold(n_splits=3,
                        #                        shuffle=True,
                        #                        random_state=1)
                        #X = np.zeros((len(train_indices_0), 2))
                        #y = labels[train_indices_0]
                        #split_1 = skf_1.split(X, y)

                        #accuracyRF = []
                        #for train_indices_1, test_indices_1 in split_1:
                        #    train_indices_1 = train_indices_0[train_indices_1]
                        #    test_indices_1 = train_indices_0[test_indices_1]
                        label_train = labels[train_indices_0]
                        label_test = labels[val]
                        FeaturesTrain = Features[train_indices_0]
                        FeaturesTest = Features[val]
                        clfRF = RandomForestClassifier(n_estimators=500,
                                                       max_depth=20)
                        #clfRF = SVC(C=1)
                        clfRF.fit(FeaturesTrain, label_train)
                        pred = clfRF.predict(FeaturesTest)
                        # accuracyRF.append(accuracy_score(pred, label_test))
                        accuracy_cross_val = accuracy_score(pred, label_test)

                        # print(np.mean(accuracyRF))
                        # if np.mean(accuracyRF) > max_accuracy_test:
                        #     max_accuracy_test = np.mean(accuracyRF)
                        #     label_train = labels[train_indices_0]
                        #     FeaturesTrain = Features[train_indices_0]
                        #     clfRF = RandomForestClassifier(n_estimators=500,
                        #                                    max_depth=20)
                        #     # clfRF = SVC(C=1)
                        #     clfRF.fit(FeaturesTrain, label_train)
                        #     label_val = labels[val]
                        #     FeaturesVal = Features[val]
                        #     pred = clfRF.predict(FeaturesVal)
                        #
                        #     accuracy_cross_val = accuracy_score(pred, label_val)
                        #     print(emb_space, emb_spec, t0, t1)
                        #     print(accuracy_cross_val)
                        # else:
                        #     pass
                        t_bis = time.time()
                        print("time for one fold : {}".format(t_bis - t))
                        accuraciesVal.append(accuracy_cross_val)
                        print(" ")
                        print("-----------------")
                        print("acc : {}".format(accuracy_cross_val))
                        print("-----------------")
                        print(" ")

                        np.save("results/" + name + str(emb_space) + str(emb_spec) + str(t0) + str(t1) + ".npy",
                                np.array(accuraciesVal))