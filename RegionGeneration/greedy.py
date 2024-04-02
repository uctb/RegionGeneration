import numpy as np
import networkx as nx
import random
import pandas as pd
from dateutil.parser import parse
from statsmodels.tsa.stattools import acf
from metrics import mape,avgACF,avgCoverage,Recall,calculate_acf
from copy import deepcopy
import folium
import time
import pickle as pkl
from scale_estimate import scale_estimate
import sys
class GreedyExp():
    '''
    #TODO: 输入、输出是要去思考和规定好的
    #! input: D矩阵 A矩阵 后续无了 
    '''
    def __init__(self,graph,demand_series,output_cluster_num) -> None:
        self.D = demand_series
        self.M = output_cluster_num
        self.bestacf = -1
        self.seed = 0
        self.graph = graph
        self.N = nx.number_of_nodes(self.graph)
        self.tried = 0
        pass
    def setupExp(self):
        self.acf_vec = np.zeros([self.M,])
        self.seed = int(time.time())%100000

    def initializeSolution(self):
        random.seed(self.seed)
        output_result = []
        condition = False
        while not condition:
            condition = True
            self.occupied_node_list = random.sample(list(nx.nodes(self.graph)),self.M)
            output_result = list(map(lambda x:[x],self.occupied_node_list))

        for ind,cluster in enumerate(output_result):
            self.acf_vec[ind] = calculate_acf(self.D,cluster,96)
        return output_result
    def _calculate_cluster_area(self,cluster):
        #type:(GreedyExp,list)->(float)
        return np.sum(self.total_area_vec[cluster])
    def getacfvec(self):
        return self.acf_vec
    def MainAlgorithm(self):
        '''
        初始化解的过程就是随机按照输出结果数从N个节点中随机挑出M个作为初始簇
            然后首先从与我们相邻的element中选出使合并后delta acf最大的element之后进行合并
        '''

        self.setupExp()
        selection = self.initializeSolution()
        istherechanged = True
        while istherechanged:
            istherechanged = False
            for index,cluster in enumerate(selection):
                neighbor_set = set()
                for node in cluster:
                    neighbor_set = neighbor_set.union(set(nx.neighbors(self.graph,node)))
                neighbor_set = neighbor_set.difference(set(self.occupied_node_list))
                max_delta_acf = -np.inf
                selected_node = -1
                selected_index = -1
                for node in neighbor_set:
                    cluster.append(node)
                    delta_acf = calculate_acf(self.D,cluster,96)-self.acf_vec[index]
                    if delta_acf > max_delta_acf:
                        max_delta_acf = delta_acf
                        selected_node = node
                        selected_index = index
                    cluster.remove(node)
                if selected_node != -1:
                    istherechanged = True
                    cluster.append(selected_node)
                    self.occupied_node_list.append(selected_node)
                    self.acf_vec[selected_index] += max_delta_acf
        return selection



if __name__ == '__main__':
    stmatrix = np.load('processed_data/processed_stmatrix.npy')
    stmatrix = stmatrix[:-96*2-2016,:]
    am = np.load('processed_data/processed_Chicago_am.npy')
    area_array = np.load('processed_data/processed_area_info.npy')
    graph = nx.Graph(am)
    best_area = 0
    best_demand = 0
    best_acf = 0
    with open('initial_result/partition.pkl','rb') as fp:
        partition = pkl.load(fp)
    acf_list = []
    for index in range(1,51):
        flag = True
        final_result = []
        for K,component in partition:
            subgraph = nx.subgraph(graph,component)
            greedyexp = GreedyExp(subgraph,stmatrix,K)
            greedyexp.setupExp()
            output_result = greedyexp.MainAlgorithm()
            final_result.append(output_result)
        node_which_cluster = [-1 for  i in range(161)]
        for component in final_result:
            for ind,cluster in enumerate(component):
                subgraph = nx.subgraph(graph,cluster)
                if not nx.is_connected(subgraph):
                    flag = False
                for element in cluster:
                    node_which_cluster[element] = ind
        for label in node_which_cluster:
            if label == -1:
                flag = False
        print(len(final_result))
        if flag:
            with open('initial_result/greedy/{}.pkl'.format(index),'wb') as fp:
                pkl.dump(final_result,fp)
        
    # print('平均coverage：{}'.format(avgcoverage))