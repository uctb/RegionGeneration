import numpy as np
import networkx as nx
import pickle as pkl
from statsmodels.tsa.stattools import acf

def scale_estimate(demand_series,area_array,graph,threshold_area,threshold_demand,threshold_acf):
    total_node_set = set(nx.nodes(graph))
    special_case_list_for_area = area_restrict(area_array,graph,threshold_area)
    special_case_list_for_demand = demand_restrict(demand_series,graph,threshold_demand)
    special_case_list_for_acf = acf_restrict(demand_series,graph,threshold_acf)
    area_set = set(special_case_list_for_area)
    demand_set = set(special_case_list_for_demand)
    acf_set = set(special_case_list_for_acf)
    special_case_set = area_set.union(demand_set).union(acf_set)
    rest_node_set = total_node_set.difference(special_case_set) 
    connected_components = nx.connected_components(nx.subgraph(graph,rest_node_set))
    return connected_components,list(map(lambda x:[x],special_case_set))

def area_restrict(area_array,graph,threshold):
    special_case_list = []
    for node in nx.nodes(graph):
        if area_array[node] > threshold:
            special_case_list.append(node)
    return special_case_list

def demand_restrict(demand_series,graph,threshold):
    special_case_list = []
    for node in nx.nodes(graph):
        if np.sum(demand_series[:,node]) > threshold:
            special_case_list.append(node)
    return special_case_list

def acf_restrict(demand_series,graph,threshold):
    special_case_list = []
    for node in nx.nodes(graph):
        if acf(demand_series[:,node],nlags=96)[95] > threshold:
            special_case_list.append(node)
    return special_case_list

if __name__=='__main__':
    # 给metis做结果
    output_cluster_num = 86
    am = np.load('processed_data/processed_Chicago_am.npy')
    demand_series = np.load('processed_data/processed_stmatrix.npy')
    graph = nx.Graph(am)
    area_array = np.load('processed_data/processed_area_info.npy')
    split_part_num_list = []
    max_region_size = 40000
    special_node_list = scale_estimate(demand_series,area_array,graph,200000,212*96)
    tmp_node_list = list(set(nx.nodes(graph)).difference(set(special_node_list)))
    tmp_graph = nx.subgraph(graph,tmp_node_list)
    cc = nx.connected_components(tmp_graph)
    for index,nodelist in enumerate(cc):
        if len(nodelist) > 1:
            split_part_num_list.append(min(np.sum(area_array[nodelist])//max_region_size+1,len(nodelist)))
            subgraph = nx.subgraph(graph,nodelist)
            with open('metis_input_graph/input_subgraph_'+str(index)+'.txt','w') as f:
                f.write(str(nx.number_of_nodes(subgraph))+' '+str(nx.number_of_edges(subgraph))+' 010'+' 2')
                for node in nodelist:
                    f.write('\n')
                    neighbor_nodes = nx.neighbors(subgraph,node)
                    f.write(str(area_array[node])+' '+str(int(np.sum(demand_series[:,node]))))
                    for neighbor_node in neighbor_nodes:
                        f.write(' '+str(nodelist.index(neighbor_node)+1))
    print(sum(split_part_num_list)+nx.number_of_nodes(graph)-len(split_part_num_list))
                 
                
                