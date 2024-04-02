
from sklearn.cluster import KMeans
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'


class BSC():
    def __init__(self,locations,K_1,demand_series,output_cluster_num):
        self.locations = locations
        self.K_1 = K_1
        self.n = self.locations.shape[0]
        self.D = demand_series
        self.output_cluster_num = output_cluster_num
        pass
    def initialcluster(self):
        k_means = KMeans(n_clusters =self.K_1, n_init=10)
        k_means.fit(self.locations)
        self.cluster_result = [[] for i in range(self.K_1)]
        for index,label in enumerate(k_means.labels_.tolist()):
            self.cluster_result[label].append(index)

    def generate_transition_matrix(self,vector_len,cluster):
        transition_matrix = []
        demand_series_all = self.D[:,cluster]
        for station in range(demand_series_all.shape[1]):
            tmp_demand_series = demand_series_all[:,station]
            transition_matrix.append(np.mean(np.array([tmp_demand_series[i*vector_len:(i+1)*vector_len] for i in range(tmp_demand_series.shape[0]//vector_len)]),axis=0,keepdims=False))
        return np.array(transition_matrix)
    def mainAlgorithm(self):
        self.initialcluster()
        cluster_result = []
        for cluster in self.cluster_result:
            k_nums = max(1,int(self.output_cluster_num/self.n*len(cluster))+1)
            tmp_cluster_result = [[] for i in range(k_nums)]
            tm_list = self.generate_transition_matrix(96,cluster)
            k_means = KMeans(n_clusters=k_nums)
            k_means.fit(tm_list)
            for index,label in enumerate(k_means.labels_.tolist()):
                tmp_cluster_result[label].append(cluster[index])
            cluster_result.extend(tmp_cluster_result)
            
        return cluster_result
    

if __name__ == '__main__':
    demand_series = np.load('processed_data/processed_stmatrix.npy')
    demand_series = demand_series[:18144,:] 
    am= np.load('processed_data/processed_Chicago_am.npy')
    area_array = np.load('processed_data/processed_area_info.npy')
    #TODO: 得到区域中心经纬度，并以此作为各location向量各维度的点
    locations = []
    for i in range(1,am.shape[0]+1):
        X = np.loadtxt('processed_data/Geocoords/'+str(i)+'.txt')
        locations.append(np.mean(X,axis=0))
    bsc = BSC(np.array(locations),50,demand_series,80)
    result = bsc.mainAlgorithm()
    demand_series = np.load('processed_data/processed_stmatrix.npy')
    from metrics import avgACF
    print(avgACF(demand_series,result))
    print(len(result))
    # import pickle as pkl
    # with open('bsc_cluster_result.pkl','wb') as fp:
    #     pkl.dump(result,fp)

