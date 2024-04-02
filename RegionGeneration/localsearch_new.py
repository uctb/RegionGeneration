from matplotlib import pyplot as plt
import numpy as np
import random
import time
import networkx as nx
import pickle as pkl
from copy import copy, deepcopy
from metrics import avgACF, calculate_acf
from scale_estimate import scale_estimate
import sys
import os 

class LocalSearchExp():
    def __init__(self,demand_series,graph,maxium_epochs,area_vec) -> None:
        super().__init__()
        self.D = demand_series
        self.Eps = maxium_epochs
        self.av = area_vec
        self.seed = 0
        self.graph = graph
        self.N = self.D.shape[1]


    pass
    def setupExp(self,dirpath):
            self.best_acf = -1
            self.best_output_result = None
            self.exit_flag = False
            self.duration_list = []
            self.current_epoch = 0
    def get_duration(self):
        return self.duration_list
    def _calculate_cluster_area(self,cluster):
        #type:(LocalSearchExp,list)->(float)
        return np.sum(self.av[cluster])
    def _generate_initial_solution(self,initial_result):
        # 如何得到一个初始的可行解
        #TODO：下午做两件事
        #TODO：  1.捋清这个逻辑就可以了，并发给学长进行评估
        #TODO：  2.捋清find movable boundary的逻辑并书写代码       
        '''
            根据total_area_vec和簇最大面积限制得到一个下限簇数。
            得到一个随机的簇数直到满足下限限制
            将连通域集按照连通域的size排序
            比较簇数和连通域数大小
                当该簇数小于等于连通域数时，顺次选择连通域作为初始解簇
                当该簇数大于连通域数时，然后依次从连通域集中选择连通域，计算连通域的面积，然后除以最大面积数作为能分成几个连通分量的限制，之后调用函数搞一下
        '''
        
        best_start_acf = -1
        best_start_output_result = initial_result
        acf_vec = []
        node_which_cluster = [-1 for i in range(self.N)]
        for index,cluster in enumerate(best_start_output_result):
            acf_vec.append(calculate_acf(self.D,cluster,96))
            for node in cluster:
                node_which_cluster[node] = index
        best_start_acf = np.mean(np.array(acf_vec))
        selection = best_start_output_result
        if best_start_acf > self.best_acf:
            self.best_acf = best_start_acf
            self.best_output_result = {'nodewhichcluster':node_which_cluster,'result':selection,'acf_vec':np.array(acf_vec)}
        return {'nodewhichcluster':node_which_cluster,'result':selection,'acf_vec':np.array(acf_vec)}
        # N = np.shape(self.A)[0]
        
        # combinations = list(it.combinations([i for i in range(0,N)],self.M))
        # com_index = 0
        
        # random.shuffle(combinations)
        # pointer = 0
        # free_node_list = [i for i in range(0,N)]
        # cluster = []
        # area_vec = np.zeros([self.M,])
        # try:
        #     tmp_result = list(combinations[com_index])
        # except:
        #     print('Can not find a solution meet the constraint')
        # com_index += 1
        # for element in tmp_result:
        #     cluster.append([element])
        #     free_node_list.remove(element)
        # while len(free_node_list)>0:
        #     if pointer < self.M:
        #         cluster_size = len(cluster[pointer])
        #         neighbor_vec = np.zeros([N,cluster_size])
        #         for i in range(cluster_size):
        #             neighbor_vec[cluster[pointer][i],i] = 1
        #         neighbor_vec = self.A @ neighbor_vec
        #         neighbor_vec = np.mean(neighbor_vec,axis = 1,keepdims=False)
        #         neighbor_list = list(np.nonzero(neighbor_vec)[0])
        #         if len(neighbor_list) > 0:
        #             selected_node = random.choice(neighbor_list)
        #         else:
        #             pointer += 1
        #             continue
        #         if area_vec[pointer] + self.av[selected_node] > self.L:
        #             pointer += 1
        #             continue
        #         else:
        #             cluster[pointer].append(selected_node)
        #             if selected_node in free_node_list:
        #                 free_node_list.remove(selected_node)
        #     else:
        #         pointer = 0
        #         free_node_list = [i for i in range(0,N)]
        #         cluster = []
        #         area_vec = np.zeros([self.M,])
        #         try:
        #             tmp_result = list(combinations[com_index])
        #         except:
        #             print('Can not find a solution meet the constraint')
        #             break
        #         com_index += 1
        #         for index in range(self.M):
        #             cluster.append([tmp_result[index]])
        #             free_node_list.remove(tmp_result[index])
        #             area_vec[index] = tmp_result[index]
        # return cluster
    

    def _get_movable_boundary(self,selection):
        # type:(LocalSearchExp,list)->list
        movable_boundary = []
        cluster_set = selection['result']
        node_which_cluster = selection['nodewhichcluster']
        for cluster in cluster_set:
            for node_u in cluster:
                neighbor_nodes_list = nx.neighbors(self.graph,node_u)
                for node_v in neighbor_nodes_list:
                    cluster_index_u = node_which_cluster[node_u]
                    cluster_index_v = node_which_cluster[node_v]
                    if cluster_index_u!=cluster_index_v:
                        tmp_cluster_v = deepcopy(cluster_set[cluster_index_v])
                        tmp_cluster_u = deepcopy(cluster_set[cluster_index_u])
                        tmp_cluster_v.append(node_u)
                        tmp_cluster_u.remove(node_u)
                        if len(tmp_cluster_u)!=0:
                            if nx.is_connected(nx.subgraph(self.graph,tmp_cluster_u)):
                                movable_boundary.append((node_u,node_v))
        return movable_boundary

    def _get_new_solution_from_neighbor(self,selection,pair):
        # type:(LocalSearchExp,list,tuple)-> None
        cluster_set = deepcopy(selection['result'])
        node_which_cluster = deepcopy(selection['nodewhichcluster'])
        acf_vec = deepcopy(selection['acf_vec'])
        node_u = pair[0]
        node_v = pair[1]
        cluster_u = node_which_cluster[node_u]
        cluster_v = node_which_cluster[node_v]
        cluster_set[cluster_u].remove(node_u)
        cluster_set[cluster_v].append(node_u)
        node_which_cluster[node_u] = node_which_cluster[node_v]
        acf_vec[cluster_u] = calculate_acf(self.D,cluster_set[cluster_u],96)
        acf_vec[cluster_v] = calculate_acf(self.D,cluster_set[cluster_v],96)
        return {'nodewhichcluster':node_which_cluster, 'result':cluster_set,'acf_vec':acf_vec}

    def mainAlgorithm(self,initial_result,dir_path):
        # type: (LocalSearchExp,list,str) -> list
        self.setupExp(dir_path)
        ls_state_dict = dict()
        while self.current_epoch<self.Eps:
            if self.exit_flag:
                break
            self.exit_flag = True
            t_1 = time.time()
            if  self.best_output_result == None:
                selection = self._generate_initial_solution(initial_result=initial_result)
            else:
                selection = self.best_output_result
            movable_boundary = self._get_movable_boundary(selection)
            # for pair in movable_boundary:
            #     solution= self._get_new_solution_from_neighbor(selection,pair)
            #     acf_vec = []
            #     for cluster in solution['result']:
            #         tmpacf = calculate_acf(self.D,cluster,96)
            #         if tmpacf == None:
            #             continue
            #         acf_vec.append(tmpacf)
            #     avgacf = np.mean(np.array(acf_vec))
            #     if avgacf > self.best_acf:
            #         self.best_acf = avgacf
            #         self.current_output_set.append(solution)
            max_delta_acf = 0
            for pair in movable_boundary:
                node_u,node_v = pair
                solution= self._get_new_solution_from_neighbor(selection,pair)
                acf_vec_new = solution['acf_vec']
                node_which_cluster_old = selection['nodewhichcluster']
                acf_vec_old = selection['acf_vec']

                tmp_delta_acf = acf_vec_new[node_which_cluster_old[node_u]]+acf_vec_new[node_which_cluster_old[node_v]] - (acf_vec_old[node_which_cluster_old[node_u]] + acf_vec_old[node_which_cluster_old[node_v]])
                if tmp_delta_acf > max_delta_acf:
                    max_delta_acf = tmp_delta_acf
                    self.best_acf = np.mean(acf_vec_new)
                    self.best_output_result =  solution
                    self.exit_flag = False
            t_2 = time.time()
            print('Epoch {}: Best acf is {};Duration is {}s'.format(self.current_epoch+1,self.best_acf,(t_2-t_1)))
            self.current_epoch += 1
        
        # TODO:结束之后应该要保存中间结果例如：收敛轮次，时间，该连通分量对应的element数和输出簇数和最终acf等

        return self.best_output_result['result']

if __name__ == '__main__':
    
    demand_series = np.load('processed_data/processed_stmatrix.npy')
    demand_series = demand_series[:-2016-96*2,:]
    adjacent_matrix = np.load('processed_data/processed_Chicago_am.npy')
    area_vec = np.load('processed_data/processed_area_info.npy')
    graph = nx.Graph(adjacent_matrix)
    '''
    with open('./initial_result/partition.pkl','rb') as fp:
        partition = pkl.load(fp)
    initial_acf_greedy = []
    converge_acf_greedy = []
    initial_acf_fluid = []
    converge_acf_fluid = []
    dir_name_list = ['async_fluid','greedy']
    for i in range(len(dir_name_list)):
        str = dir_name_list[i]
        dirpath = os.path.join('initial_result',str)
        for file in os.listdir(dirpath):
            final_output_result = []
            final_initial_result = []
            with open(os.path.join(dirpath,file),'rb') as fp:
                initial_result = pkl.load(fp)
            for subpartition,sub_initial_result in zip(partition,initial_result):
                final_initial_result.extend(sub_initial_result)
                nodelist = subpartition[1]
                subgraph = nx.subgraph(graph,nodelist)
                lsexp = LocalSearchExp(demand_series=demand_series,graph=subgraph,maxium_epochs=30,area_vec=area_vec)
                final_output_result.extend(lsexp.mainAlgorithm(initial_result=sub_initial_result,dir_path='./model'))
            if i == 0:
                initial_acf_fluid.append(avgACF(demand_series,final_initial_result))
                converge_acf_fluid.append(avgACF(demand_series,final_output_result))
            if i == 1:
                initial_acf_greedy.append(avgACF(demand_series,final_initial_result))
                converge_acf_greedy.append(avgACF(demand_series,final_output_result))
        
    with open('./initial_result/partition.pkl','rb') as fp:
        partition = pkl.load(fp)
    dirpath = os.path.join('initial_result','metis')
    initial_acf_metis = []
    converge_acf_metis = []
    for file in os.listdir(dirpath):
        final_output_result = []
        with open(os.path.join(dirpath,file),'rb') as fp:
            initial_result = pkl.load(fp)
            tmp_initial_result = [initial_result[64],initial_result[65],initial_result[66],initial_result[67],initial_result[68]]
            subgraph = nx.subgraph(graph,partition[3][1])
            lsexp = LocalSearchExp(demand_series=demand_series,graph=subgraph,maxium_epochs=1000,area_vec=area_vec)
            final_output_result.extend(lsexp.mainAlgorithm(initial_result=tmp_initial_result,dir_path='./model'))
        final_output_result.extend(initial_result[:-5])
        final_initial_result = initial_result
        initial_acf_metis.append(avgACF(demand_series,final_initial_result))
        converge_acf_metis.append(avgACF(demand_series,final_output_result))
    '''
#     font_dict = {
#     'size':14,
#     'family':'Times New Roman',
#     'weight':'bold'
# }
    import pandas as pd
    import seaborn as sns

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['font.serif'] = 'Times New Roman'
   
    sns.set_theme(style='darkgrid',font='Times New Roman',font_scale=1.22,palette='husl')
    initial_acf_metis = np.load('./initial_result/initial_acf_metis.npy')
    converge_acf_metis = np.load('./initial_result/converge_acf_metis.npy')
    initial_acf_fluid = np.load('./initial_result/initial_acf_fluid.npy')
    converge_acf_fluid = np.load('./initial_result/converge_acf_fluid.npy')
    initial_acf_greedy = np.load('./initial_result/initial_acf_greedy.npy')
    converge_acf_greedy = np.load('./initial_result/converge_acf_greedy.npy')

    # plt.scatter(1.5*np.array(initial_acf_fluid),1.5*np.array(converge_acf_fluid),color='r',marker='s')
    # plt.scatter(1.5*np.array(initial_acf_greedy),1.5*np.array(converge_acf_greedy),color='b',marker='x')
    # plt.scatter(1.5*np.array(initial_acf_metis),1.5*np.array(converge_acf_metis),color='g',marker='o')
    # plt.legend(['fluid','greedy','metis'])
    # plt.xlabel('Initial acf_daily',fontdict=font_dict)
    # plt.ylabel('Convergent acf_daily',fontdict=font_dict)
    # plt.tick_params(labelsize=12,width=3)
    # plt.savefig('./initial.png',dpi=600)
    method = []
    for i in range(initial_acf_greedy.shape[0]):
        method.append('Greedy')
    for i in range(initial_acf_fluid.shape[0]):
        method.append('Fluid')
    for i in range(initial_acf_metis.shape[0]):
        method.append('D-Balance')
    data = {
        'init acf':np.concatenate((initial_acf_greedy,initial_acf_fluid,initial_acf_metis)),
        'last acf':np.concatenate((converge_acf_greedy,converge_acf_fluid,converge_acf_metis)),
        'method':method
    }
    sns.regplot(x='init acf',y='last acf',data=pd.DataFrame(data),truncate=True,scatter=False,color='darkblue')
    sns.scatterplot(x='init acf',y='last acf',data=pd.DataFrame(data),hue='method',style='method',s=80)
    plt.savefig('./initial.png',dpi=600)
    plt.show()

   