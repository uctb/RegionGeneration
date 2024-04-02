import networkx as nx
import numpy as np
import pickle as pkl

with open('initial_result/partition.pkl','rb') as fp:
    partition = pkl.load(fp)

stmatrix = np.load('processed_data/processed_stmatrix.npy')
stmatrix = stmatrix[:-96*2-2016,:]
am = np.load('processed_data/processed_Chicago_am.npy')
area_array = np.load('processed_data/processed_area_info.npy')
graph = nx.Graph(am)
for ind in range(1,51):
    final_result = []
    flag = True
    for K,component in partition:
        subgraph = nx.subgraph(graph,component)
        isconditionmet = False
        K = int(K)
        result = nx.algorithms.community.asyn_fluidc(subgraph,K,seed=K+ind)
        result = list(map(lambda x:list(x),result))
        final_result.append(result)
    node_which_cluster = [-1 for  i in range(161)]
    for component in final_result:
            for index,cluster in enumerate(component):
                subgraph = nx.subgraph(graph,cluster)
                if not nx.is_connected(subgraph):
                    flag = False
                for element in cluster:
                    node_which_cluster[element] = index
    for label in node_which_cluster:
        if label == -1:
            flag = False
    if flag:
            print(len(final_result))

            with open('initial_result/async_fluid/{}.pkl'.format(ind),'wb') as fp:
                pkl.dump(final_result,fp)


