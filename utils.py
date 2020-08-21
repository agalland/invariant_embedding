import networkx as nx
import numpy as np
import multiprocessing
import scipy.sparse as sp

from joblib import Parallel, delayed
from mlxtend.data import loadlocal_mnist
from numpy import linalg as LA
from sklearn.preprocessing import LabelEncoder


class GraphDataset:
    def __init__(self, folder_path='', with_node_features=False, n_graphs=1000):

        if folder_path.split("/")[-2] != "mnist":
            g = nx.Graph()
            data_adj = np.loadtxt(folder_path + '_A.txt', delimiter=',').astype(int)
            data_graph_indicator = np.loadtxt(folder_path + '_graph_indicator.txt',
                                              delimiter=',').astype(int)
            labels = np.loadtxt(folder_path + '_graph_labels.txt',
                                delimiter=',').astype(int)
            # If features aren't available, compute one-hot degree vectors
            try:
                node_labels = np.loadtxt(folder_path + '_node_labels.txt', delimiter=',').astype(int)
                node_labels -= np.min(node_labels)
                max_feat = np.max(node_labels) + 1
                mat_feat = np.eye(max_feat)
                with_node_features = True
            except:
                with_node_features = False

            data_tuple = list(map(tuple, data_adj))
            g.add_edges_from(data_tuple)
            g.remove_nodes_from(list(nx.isolates(g)))

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
                g_sub = g.subgraph(nodes).copy()

                max_cc = max(nx.connected_components(g_sub), key=len)
                g_sub = g_sub.subgraph(max_cc).copy()

                adj = np.array(nx.adjacency_matrix(g_sub).todense())
                self.degree_max = max(self.degree_max, np.max(np.sum(adj, 0)))
                nodes = range(len(adj))
                g_sub.graph['label'] = self.labels_[i]
                nx.convert_node_labels_to_integers(g_sub)

                tmp = len(nodes)
                self.graphs_.append(g_sub)
                if tmp > max_num_nodes:
                    max_num_nodes = tmp

                if with_node_features:
                    nodes = list(g_sub.nodes()) - np.min(list(g.nodes()))
                    feat_index = node_labels[nodes]
                    node_feat = mat_feat[feat_index]
                    self.node_features.append(node_feat)

            if not with_node_features:
                mat_feat = np.eye(self.degree_max+1)
                for i in range(graph_num):
                    g_sub = self.graphs_[i]
                    deg = np.array(list(dict(nx.degree(g_sub)).values()))
                    node_feat = mat_feat[deg]
                    self.node_features.append(node_feat)

            self.n_graphs_ = len(self.graphs_)
            self.graphs_ = np.array(self.graphs_)
            self.max_num_nodes = max_num_nodes

            print('Loaded', self.n_graphs_, self.max_num_nodes)
        else:
            self.n_graphs = n_graphs
            X_mnist, y_mnist = loadlocal_mnist(
                images_path=folder_path+'/train-images-idx3-ubyte.gz',
                labels_path=folder_path+'/train-labels-idx1-ubyte')
            print("computing graphs")
            X, y = training_set_mnist(X_mnist, y_mnist, n_graphs, folder_path)
            self.graphs_ = [X[k][0] for k in range(n_graphs)]
            self.node_features = [X[k][1] for k in range(n_graphs)]
            self.max_num_nodes = len(X[0][0])
            self.labels_ = y
            self.n_graphs_ = len(self.graphs_)

            print('Loaded', self.n_graphs_, self.max_num_nodes)


def aggregate(x, t_min, t_max, nbins):
    t = np.linspace(t_min, t_max, nbins).reshape(-1, 1)
    out = None
    for k in range(len(x)):
        x_k = x[k]
        maxk = np.max(x_k)
        expx = np.exp(t * (x_k - maxk))
        if out is None:
            out = np.sum(x_k * expx, 1) / np.sum(expx, 1)
        else:
            out = np.concatenate((out, np.sum(x_k * expx, 1) / np.sum(expx, 1)))

    return out


def hist_invar(x, t1, range_min=-1., range_max=1.):
    r = (range_min, range_max)
    out = None
    for k in range(len(x)):
        hist = np.histogram(x[k], range=r, bins=t1)[0]
        if out is None:
            out = hist
        else:
            out = np.concatenate((out, hist))

    return out


def padded_embedding(graph, emb_space=10, emb_spectral=20, max_nodes=30, feat=None, t=0):
    A = np.asarray(nx.to_numpy_matrix(graph))
    n_nodes, m_nodes = A.shape
    k = min(emb_spectral, n_nodes - 1)
    embedding = np.zeros((emb_spectral, n_nodes))
    L = np.diag(np.sum(A, axis=0)) - A
    eigenvalues, eigenvectors = LA.eigh(L)
    indices = np.argsort(eigenvalues)[1:k + 1]

    eigvecNorm = np.dot(np.diag(np.sqrt(1 / eigenvalues[1:])),
                        eigenvectors.T[1:, :])
    sim = np.dot(eigvecNorm.T, eigvecNorm)
    sim = np.reshape(sim, (1, -1))
    hist = np.histogram(sim, range=(-1, 1), bins=t)[0]

    embedding[:k, :] = np.dot(np.diag(np.sqrt(1 / eigenvalues[indices])),
                              eigenvectors.T[indices, :])

    emb_space_full = emb_space
    if feat is not None:
        feat = np.array(feat)
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
    return np.vstack((embed_space,embedding)),\
           np.pad(eigenvalues[indices], (max_nodes-len(indices),0), 'constant'),\
           hist


def dataset_embedding(dataset, emb_space, emb_spectral, max_num_nodes, t0, t1):

    list_graphs = []
    for i, g in enumerate(dataset.graphs_):
        feat = dataset.node_features[i]
        #feat = None
        # if i % 500 == 0:
        #     print("{}%".format(round(i * 100 / len(dataset.graphs_)), 3))

        x, eig, hist = padded_embedding(g,
                                        emb_space,
                                        emb_spectral,
                                        max_num_nodes,
                                        feat=feat,
                                        t=t0)

        x_m = 0.5 * (aggregate(x, 0, 1, t1) + aggregate(-x, 0, 1, t1))
        #x_m = np.mean(x, 1)
        #x_m = hist_invar(x, t1)
        y = dataset.labels_[i]
        concatenate = np.concatenate((x_m, eig, hist))
        list_graphs.append((concatenate, y))

    return list_graphs


def training_set_mnist(X_mnist, y_mnist, num_images_train, data_path):
    num_cores = multiprocessing.cpu_count()
    while True:
        try:
            print("loading graphs")
            Xy = Parallel(n_jobs=num_cores)(delayed(load_graph)(y_mnist, im_i, data_path) \
                                        for im_i, im in enumerate(X_mnist[:num_images_train]))
            X = [(Xy[k][0], Xy[k][1]) for k in range(len(Xy))]
            y = np.array([Xy[k][2] for k in range(len(Xy))])
            break
        except:
            print("matrices not computed")
            print("computing matrices")
            Parallel(n_jobs=num_cores)(delayed(process_graph2image)(i, im, data_path) \
                                        for i, im in enumerate(X_mnist[:num_images_train]))

    return X, y


def load_graph(y_mnist, im_i, data_path):
    adj = sp.load_npz(data_path + "/images/" + "image" + str(im_i) + ".npz")
    adj = np.array(adj.todense())
    adj = adj * (adj > 1e-6)
    adj = sp.csr_matrix(adj)
    deg = np.diag(np.array(adj.sum(0))[0])
    g = nx.from_scipy_sparse_matrix(adj)

    nodeFeature = sp.load_npz(
        data_path + "/images/" + "feature" + str(im_i) + ".npz")
    y = y_mnist[im_i]
    # adj = normalize(adj + sp.eye(adj.shape[0]), norm="l1", axis=1)
    return g, nodeFeature, y


# Transform image into graph
def distance_intensity(im, node1, node2, sigma=0.01):
    if len(im.shape) > 2:
        val = float(np.exp(- 1 * np.linalg.norm(im[node1[0]][node1[1]] - im[node2[0]][node2[1]], ord=1) / sigma))
    else:
        val = float(np.exp(- 1 * np.abs(float(im[node1[0]][node1[1]]) - float(im[node2[0]][node2[1]])) / sigma))
    return val


def distance_space(pos1, pos2, sigma=5.):
    val = float(np.exp(- np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) / sigma))
    return val


def image2graph(im, dis_max=4, sigma=0.01):
    G = nx.Graph()
    nodeFeature = np.zeros((im.shape[0] * im.shape[1], 256))
    for node in range(im.shape[0] * im.shape[1]):
        G.add_node(node)

    dis_array = [0]
    for k in range(1,dis_max):
        dis_array.append(-k)
        dis_array.append(k)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            nodeFeature[im.shape[1]*i+j][im[i][j]] = 1.
            pos1 = [i,j]
            for ki in range(dis_max):
                for kj in dis_array:
                    if i+ki < im.shape[0]:
                        if j+kj > -1 and j+kj < im.shape[1]:
                            pos2 = [i+ki, j+kj]
                            if pos1 != pos2:
                                if np.sqrt(ki**2 + kj**2) < dis_max:
                                    dist_intensity = distance_intensity(im, pos1, pos2, sigma)
                                    dist_space = distance_space(pos1, pos2)
                                    w = dist_intensity * dist_space
                                    #print(i, j, ki+i, kj+j, im[i,j], im[ki+i, kj+j], dist_intensity)
                                    G.add_weighted_edges_from([(im.shape[1]*i+j, im.shape[1]*(i+ki)+j+kj, w)])
    return G, nodeFeature


def process_graph2image(i, im, data_path):
    im = np.array(im).reshape(28, 28)
    g, nodeFeature = image2graph(im, dis_max=4, sigma=20.)
    nodeFeature = sp.csr_matrix(nodeFeature)
    adj = nx.adjacency_matrix(g)
    adj = np.array(adj.todense())
    adj = adj * (adj > 1e-6)
    adj = sp.csr_matrix(adj)
    sp.save_npz(data_path+"/images/" + "image" + str(i) + ".npz", adj)
    sp.save_npz(data_path+"/images/" + "feature" + str(i) + ".npz", nodeFeature)
