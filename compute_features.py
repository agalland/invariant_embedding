import networkx as nx
import numpy as np

from numpy import linalg as LA
from sklearn.preprocessing import LabelEncoder


class GraphDataset:
    def __init__(self, folder_path='', with_node_features=False):

        G = nx.Graph()
        data_adj = np.loadtxt(folder_path + '_A.txt', delimiter=',').astype(int)
        data_graph_indicator = np.loadtxt(folder_path + '_graph_indicator.txt',
                                          delimiter=',').astype(int)
        labels = np.loadtxt(folder_path + '_graph_labels.txt',
                            delimiter=',').astype(int)
        if with_node_features:
            node_labels = np.loadtxt(folder_path + '_node_labels.txt', delimiter=',').astype(int)
            node_labels -= np.min(node_labels)
            max_feat = np.max(node_labels) + 1
            mat_feat = np.eye(max_feat)

        data_tuple = list(map(tuple, data_adj))
        G.add_edges_from(data_tuple)
        G.remove_nodes_from(list(nx.isolates(G)))

        # split into graphs

        le = LabelEncoder()
        self.labels_ = le.fit_transform(labels)
        self.n_classes_ = len(le.classes_)
        self.n_graphs_ = len(self.labels_)

        graph_num = data_graph_indicator.max()
        node_list = np.arange(data_graph_indicator.shape[0]) + 1
        self.graphs_ = []
        self.node_features = []
        max_num_nodes = 0
        self.degree_max = 0
        for i in range(graph_num):
            if i % 500 == 0:
                print("{}%".format(round((i * 100) / graph_num), 3))

            nodes = node_list[data_graph_indicator == i + 1]
            G_sub = G.subgraph(nodes).copy()

            max_cc = max(nx.connected_components(G_sub), key=len)
            G_sub = G_sub.subgraph(max_cc).copy()

            A = np.array(nx.adjacency_matrix(G_sub).todense())
            self.degree_max = max(self.degree_max, np.max(np.sum(A, 0)))
            nodes = range(len(A))
            G_sub.graph['label'] = self.labels_[i]
            nx.convert_node_labels_to_integers(G_sub)

            tmp = len(nodes)
            self.graphs_.append(G_sub)
            if tmp > max_num_nodes:
                max_num_nodes = tmp

            if with_node_features:
                nodes = list(G_sub.nodes()) - np.min(list(G.nodes()))
                feat_index = node_labels[nodes]
                node_feat = mat_feat[feat_index]
                self.node_features.append(node_feat)

        self.n_graphs_ = len(self.graphs_)
        self.graphs_ = np.array(self.graphs_)
        self.max_num_nodes = max_num_nodes

        print('Loaded', self.n_graphs_, self.max_num_nodes)


def feature_eig(graph, emb_spectral):
    A = np.asarray(nx.to_numpy_matrix(graph))
    L = np.diag(np.sum(A, axis=0)) - A
    eigenvalues, eigenvectors = LA.eigh(L)
    emb_eig = np.zeros(emb_spectral)
    min_dim = min(len(eigenvalues), emb_spectral)
    emb_eig[-min_dim:] = eigenvalues[:min_dim]

    return emb_eig

def feature_hist(graph, t):
    A = np.asarray(nx.to_numpy_matrix(graph))
    L = np.diag(np.sum(A, axis=0)) - A
    eigenvalues, eigenvectors = LA.eigh(L)

    eigvecNorm = np.dot(np.diag(np.sqrt(1 / eigenvalues[1:])),
                        eigenvectors.T[1:, :])
    sim = np.dot(eigvecNorm.T, eigvecNorm)
    sim = np.reshape(sim, (1, -1))
    hist = np.histogram(sim, range=(-1, 1), bins=t)[0]

    return hist

def feature_spectrale(graph, emb_spectral):
    A = np.asarray(nx.to_numpy_matrix(graph))
    n_nodes, m_nodes = A.shape
    k = min(emb_spectral, n_nodes - 1)
    embedding = np.zeros((emb_spectral, n_nodes))
    L = np.diag(np.sum(A, axis=0)) - A
    eigenvalues, eigenvectors = LA.eigh(L)
    indices = np.argsort(eigenvalues)[1:k + 1]

    embedding[:k, :] = np.dot(np.diag(np.sqrt(1 / eigenvalues[indices])),
                              eigenvectors.T[indices, :])

    return embedding

def feature_space(graph, emb_space, feat=None, max_deg=None):
    A = np.asarray(nx.to_numpy_matrix(graph))
    n_nodes, m_nodes = A.shape
    emb_space_full = emb_space
    if feat is not None:
        featdim = feat.shape[1]
        emb_space_full = emb_space * featdim
    else:
        if max_deg is not None:
            mat_eye = np.eye(max_deg + 1)
            degs = list(np.sum(A, 0).astype(int))
            feat = mat_eye[degs]
            featdim = feat.shape[1]
            emb_space_full = emb_space * featdim

    embed_space = np.zeros((emb_space_full, n_nodes))
    P = np.dot(np.eye(A.shape[0]) / np.sum(A, axis=0), A)
    Q = np.dot(np.eye(A.shape[0]) / np.sum(A, axis=0), A)
    for i in range(emb_space):
        P = np.dot(P, Q)
        if feat is None:
            embed_space[i, :] = np.dot(np.ones(A.shape[0]), P)
        else:
            embed_space[i * featdim:(i + 1) * featdim, :] = np.dot(P, feat).T

    return embed_space

def feature_wavelet(graph, emb_space, feat=None):
    A = np.asarray(nx.to_numpy_matrix(graph))
    n_nodes, m_nodes = A.shape

    deg = np.sum(A, 0)
    D_inv = np.linalg.inv(np.diag(deg) ** (0.5))

    T = 0.5 * (np.eye(A.shape[0]) + np.dot(np.dot(D_inv, A), D_inv))

    Psi_0 = np.eye(A.shape[0]) - T
    Psi = [Psi_0]
    for j in range(1, emb_space):
        Psi_j = T ** (2 ** (j - 1)) - T ** (2 ** j)
        Psi.append(Psi_j)

    emb_space_full = emb_space
    if feat is not None:
        featdim = feat.shape[1]
        emb_space_full = emb_space * featdim

    embed_space = np.zeros((emb_space_full, n_nodes))
    for i in range(emb_space):
        if feat is None:
            embed_space[i, :] = np.dot(Psi[i], np.ones(A.shape[0]))
        else:
            embed_space[i * featdim:(i + 1) * featdim, :] = np.dot(Psi[i], feat).T

    return embed_space


params = {'emb_space': [3, 5], 'emb_spec': [20, 50], "t0": [21, 31]}
#params = {'emb_space': [1, 2, 3, 4, 5], 'emb_spec': [2, 3, 5, 8, 10, 20], 'max_num_nodes': [2, 3, 5, 8, 10, 20], "t0": [11, 21, 31]}
graphs_name = ["COLLAB"]
data_pwd = "/data/these/graph_classif/"
with_node_features = False

datasets = {}
for gn in graphs_name:
    datasets[gn] = GraphDataset(folder_path=data_pwd + gn + '/' + gn, with_node_features=with_node_features)

for name, dataset in datasets.items():

    print(name)

    for i, graph in enumerate(dataset.graphs_):
        print(i)
        feat = None
        if with_node_features:
            feat = dataset.node_features[i]
        print("space")
        for emb_space in params["emb_space"]:
            print(emb_space)
            feature = feature_space(graph, emb_space, feat=feat, max_deg=dataset.degree_max)
            # feature = feature_wavelet(graph, emb_space, feat)
            np.save(data_pwd + "features/space/" + name + "_" + str(i) + "_space_" + str(emb_space), feature)
        # print("spec")
        # for emb_spec in params["emb_spec"]:
        #     print(emb_spec)
        #     feature = feature_spectrale(graph, emb_spec)
        #     np.save(data_pwd + "features/spectrale/" + name + "_" + str(i) + "_spec_" + str(emb_spec), feature)
        print("hist")
        for t0 in params["t0"]:
            print(t0)
            feature = feature_hist(graph, t0)
            np.save(data_pwd + "features/histogramme/" + name + "_" + str(i) + "_hist_" + str(t0), feature)
        print("eig")
        for emb_spec in params["emb_spec"]:
            print(emb_spec)
            feature = feature_eig(graph, emb_spec)
            np.save(data_pwd + "features/eigenvalues/" + name + "_" + str(i) + "_eig_" + str(emb_spec), feature)
