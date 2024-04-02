import numpy as np
from copy import copy,deepcopy
import sys
import random
import time
from scipy.stats import pearsonr
import networkx as nx
from scale_estimate import scale_estimate
from metrics import avgACF,avgCorr
from math import radians,sin,cos,asin,sqrt

def haversine_dis(lon1, lat1, lon2, lat2):
    #将十进制转为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    #haversine公式
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    aa = sin(d_lat/2)**2 + cos(lat1)*cos(lat2)*sin(d_lon/2)**2
    c = 2 * asin(sqrt(aa))
    r = 6371 # 地球半径，千米
    return c*r*1000


class GCSC():
    def __init__(self,demand_series,graph,area_array,number_of_node_in_total_graph,locations) -> None:
        self.D = demand_series
        self.graph = graph
        self.N = number_of_node_in_total_graph
        self.area_array = area_array
        self.total_node_list = list(nx.nodes(self.graph))
        self.locations = locations
        self.tau = 1000
        self.initialresult = None

    def _calculate_cluster_area(self,cluster):
        return np.sum(self.area_array[cluster])
    def _valuefunction_1(self,node,cluster):
        tmp_sum = []
        neighbor_list = list(nx.neighbors(self.graph,node))
        for node_ in self.cluster_result[cluster]:
            if node_ in neighbor_list:
                tmp_sum.append((1+pearsonr(self.D[:,node],self.D[:,node_])[0])/2)
        if len(tmp_sum) == 0:
            return -np.inf
        return np.mean(np.array(tmp_sum))
    def _valuefunction_2(self,node,cluster):
        tmp_sum = []
        neighbor_list = list(nx.neighbors(self.graph,node))
        max_dis = 0
        for node_ in self.cluster_result[cluster]:
            if node_ in neighbor_list:
                tmp_dis = haversine_dis(self.locations[node][1],self.locations[node][0],self.locations[node_][1],self.locations[node_][0])
                if max_dis<tmp_dis:
                    max_dis = tmp_dis
                tmp_sum.append((1+pearsonr(self.D[:,node],self.D[:,node_])[0])/2)
        if len(tmp_sum) == 0:
            return -np.inf
        return np.sum(np.array(tmp_sum))*np.log(self.tau/max_dis)
    def find_adjcluster_set(self,node):
        cluster_set = []
        for node_ in nx.neighbors(self.graph,node):
            cluster_set.append(self.node_which_cluster[node_])
        return set(cluster_set)
    def initializelabels(self,expected_num,max_region_size):
        Isconditionmeeted = False
        tried = 0
        addition = 0
        while not Isconditionmeeted:
            Isconditionmeeted = True
            self.cluster_result = []
            K = expected_num+addition
            result = nx.algorithms.community.asyn_fluidc(self.graph,K,seed=int(time.time())%10000+tried)
            for community in result:
                nodelist = list(community)
                subgraph = nx.subgraph(self.graph,nodelist)
                if self._calculate_cluster_area(nodelist) > max_region_size and nx.is_connected(subgraph):
                    # if len(component)>=1:
                        # component.remove(nodelist[np.argmax(self.av[nodelist])])
                    Isconditionmeeted = False
                    break
                self.cluster_result.append(nodelist)
            if not Isconditionmeeted:
                if tried >= 20:
                    tried = 0
                    addition += 1
                else:
                    tried += 1
                continue
            addition = 0
        self.node_which_cluster = [-1 for i in range(self.N)]
        for index,cluster in enumerate(self.cluster_result):
            for node in cluster:
                self.node_which_cluster[node] = index
        return self.cluster_result
    def subsample(self,sample_num):
        node_list = random.sample(self.total_node_list,sample_num)
        return node_list
    def mainAlgorithm(self,epoch_max,expected_num,max_region_size):
        self.initialresult = deepcopy(self.initializelabels(expected_num=expected_num,max_region_size=max_region_size))
        labelshavechanged = True
        epochs = 0
        number_of_nodes = nx.number_of_nodes(self.graph)
        while labelshavechanged and epochs<= epoch_max:
            labelshavechanged = False
            last_time_node_list = copy(self.node_which_cluster)
            node_list = self.subsample(number_of_nodes)
            for node in node_list:
                cluster_set = self.find_adjcluster_set(node)
                target_label = self.node_which_cluster[node]
                last_label = target_label
                if len(self.cluster_result[target_label]) == 1:
                    continue 
                acf_throw_1 = avgACF(self.D,[self.cluster_result[target_label]])
                correlation_throw_1 = avgCorr(self.D,[self.cluster_result[target_label]])
                # print('**********************Before***********************')
                # print('acf:{};correlation:{}'.format(acf_throw_1,correlation_throw_1))
                self.cluster_result[target_label].remove(node)
                acf_throw_2 = avgACF(self.D,[self.cluster_result[target_label]])
                correlation_throw_2 = avgCorr(self.D,[self.cluster_result[target_label]])
                # print('**********************After***********************')
                # print('acf:{};correlation:{}'.format(acf_throw_2,correlation_throw_2))
                subgraph = nx.subgraph(self.graph,self.cluster_result[target_label])
                if not nx.is_connected(subgraph):
                    self.cluster_result[target_label].append(node)
                    continue
                max_value = -np.inf
                for cluster in cluster_set:
                    value = self._valuefunction_2(node,cluster)
                    tmp_cluster = copy(self.cluster_result[cluster])
                    tmp_cluster.append(node)
                    subgraph = nx.subgraph(self.graph,tmp_cluster)
                    if value>max_value and np.sum(self.area_array[tmp_cluster]) < max_region_size:# and nx.is_connected(subgraph):
                        target_label = cluster
                        max_value = value
                acf_add_1 = avgACF(self.D,[self.cluster_result[target_label]])
                correlation_add_1 = avgCorr(self.D,[self.cluster_result[target_label]])
                # print('**********************Before***********************')
                # print('acf:{};correlation:{}'.format(acf_add_1,correlation_add_1))
                self.node_which_cluster[node] = target_label
                self.cluster_result[target_label].append(node)
                acf_add_2 = avgACF(self.D,[self.cluster_result[target_label]])
                correlation_add_2 = avgCorr(self.D,[self.cluster_result[target_label]])
                # print('**********************After***********************')
                # print('acf:{};correlation:{}'.format(acf_add_2,correlation_add_2))
                # if correlation_add_2-correlation_add_1 < correlation_throw_1-correlation_throw_2-0.01:
                #     print('Node:{}',format(node))
                #     print('clusters waiting for chosen:')
                #     for cluster in cluster_set:
                #         print(self.cluster_result[cluster])
                #     print('cluster before change labels:{}'.format(self.cluster_result[last_label]))
                #     print('cluster after change labels:{}'.format(self.cluster_result[target_label]))
            
            # acf = avgACF(self.D,self.cluster_result)
            # correlation = avgCorr(self.D,self.cluster_result)
            # print('**********************After***********************')
            # print('acf:{};correlation:{}'.format(acf,correlation))
            for index in range(self.N):
                if last_time_node_list[index] != self.node_which_cluster[index]:
                    labelshavechanged = True 
            epochs+=1
        return self.cluster_result
    def getinitresult(self):
        return self.initialresult

#TODO: 找到计算两经纬度点之间距离的方法

if __name__ == '__main__':
    demand_series = np.load('processed_data/processed_stmatrix.npy')
    am= np.load('processed_data/processed_Chicago_am.npy')
    area_array = np.load('processed_data/processed_area_info.npy')
    #TODO: 得到区域中心经纬度，并以此作为各location向量各维度的点
    locations = []
    for i in range(1,am.shape[0]+1):
        X = np.loadtxt('processed_data/Geocoords/'+str(i)+'.txt')
        locations.append(np.mean(X,axis=0))
    graph = nx.Graph(am)
    t_area =56500
    t_demand = 100000000000000
    t_acf = 1
    # connected_components,special_list = scale_estimate(demand_series=demand_series,area_array=area_array,graph=graph,threshold_area=t_area,threshold_demand=t_demand,threshold_acf=t_acf)
    for i in range(2000):
        init_result = []
        final_result = []
        connected_components,special_list = scale_estimate(demand_series=demand_series,area_array=area_array,graph=graph,threshold_area=t_area,threshold_demand=t_demand,threshold_acf=t_acf)
        for ind,component in enumerate(connected_components):
            nodelist = list(component)
            if len(nodelist) == 1:
                final_result.extend([nodelist])
                init_result.extend([nodelist])
                continue
            subgraph = nx.subgraph(graph,component)
            # print('*****************************{}***************************'.format(ind))
            gcsc = GCSC(demand_series=demand_series,graph=subgraph,area_array=area_array,number_of_node_in_total_graph=nx.number_of_nodes(graph),locations=locations)
            expected_num = min(np.sum(area_array[nodelist])//t_area+1,len(nodelist))
            final_result.extend(gcsc.mainAlgorithm(100,int(expected_num),t_area))
            init_result.extend(gcsc.getinitresult())
        final_result.extend(special_list)
        init_result.extend(special_list)
        node_which_cluster = [-1 for i in range(am.shape[0])]
        for ind,cluster in enumerate(final_result):
            for node in cluster:
                node_which_cluster[node] += 1
        for label in node_which_cluster:
            if label != 0:
                print('fuck')
        node_which_cluster = [-1 for i in range(am.shape[0])]
        for ind,cluster in enumerate(final_result):
            for node in cluster:
                node_which_cluster[node] = ind
        for label in node_which_cluster:
            if label == -1:
                print('fuck')
        for i in range(len(final_result)-1,-1,-1):
            if len(final_result[i]) == 0:
                del final_result[i]
        print(len(final_result))
        print('acf:',avgACF(demand_series,final_result))
        # print(len(init_result))
        # print('acf:',avgACF(demand_series,init_result))
        if len(final_result) == 95:
            import pickle as pkl
            with open('outputresult/data/gcsc_cluster_task_final.pkl','wb') as fp:
                pkl.dump(final_result,fp)
            # with open('outputresult/data/gcsc_cluster_initial.pkl','wb') as fp:
            #     pkl.dump(init_result,fp)
            sys.exit(0)