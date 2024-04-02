import random
from shutil import which
from metrics import avgACF
from sklearn import cluster
from sklearn.cluster import dbscan
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import math
from dateutil.parser import parse
from datetime import timedelta
from make_uctb_dataset import build_uctb_dataset
import pickle as pkl


def distance(origin, destination):
    """
    (lat,lon)
    kilometers
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d



# df = pd.read_csv(path)
# df.drop_duplicates(['date','latitude','longitude'],keep='first',inplace=True)

# TODO: subsample

def subsample(counts_array,lat_only_list,lng_only_list,df,total_sample_num):
    counts_list = (counts_array/np.sum(counts_array)*total_sample_num).astype(np.int64).tolist()
    X = np.array([[],[]]).T
    node_which_cluster = []
    ind = 0
    for sample_num,lat_only,lng_only in zip(counts_list,lat_only_list,lng_only_list):
        '''
        逻辑上
        每个逻辑条件构成一个df的子集，然后按比例进行随机采样
        '''
        df_tmp = df[(df['latitude']==lat_only) & (df['longitude']==lng_only)] # 逻辑条件筛选子集
        tmp_lat_lng_list = []
        for lat,lng in zip(df_tmp['latitude'],df_tmp['longitude']):
            tmp_lat_lng_list.append([lat,lng])
        if sample_num == 0:
            sample_num = 1
        X = np.concatenate((X,np.array(random.sample(tmp_lat_lng_list,sample_num))))
        tmplist = [ind for i in range(sample_num)]
        ind += 1
        node_which_cluster.extend(tmplist)
    return X,node_which_cluster

def calculate_recall(counts_array,node_which_cluster,labels,lat_list):
    occupied_demand = 0
    label_list = [-1 for i in range((counts_array.shape[0]))]
    isadded_list = [False for i in range((counts_array.shape[0]))]
    for cluster,label in zip(node_which_cluster,labels):
        label_list[cluster] = label
        if label == -1 or isadded_list[cluster]:
            continue
        else:
            occupied_demand += counts_array[cluster]
            isadded_list[cluster] = True
    return occupied_demand/np.sum(counts_array), dict(zip(lat_list,label_list))

def make_stmatrix_and_node_station_info(df,dict_lat_to_label,cluster_num):
    stmatrix = np.zeros([20352,cluster_num])
    td = timedelta(minutes=15)
    reference_date = parse('10/01/2018 12:00:00 AM')
    flag_list = [False for i in range(cluster_num)]
    node_station_info = []
    for date,lat,lng in zip(df['date'],df['latitude'],df['longitude']):
        temporalindex = (parse(date) - reference_date)//td
        spatialindex = dict_lat_to_label[lng]
        if spatialindex == -1:
            continue
        if not flag_list[spatialindex]:
            flag_list[spatialindex] = True
            node_station_info.append([spatialindex,'2018-10-01',lat,lng,str(spatialindex)])
        stmatrix[temporalindex,spatialindex]+=1
    return stmatrix,node_station_info


if __name__ == '__main__':
    

    # path = 'E:/project/mission/didi/data/event/taxi.csv'
    # df_only = pd.read_csv('latlng.csv')
    # # X = np.transpose(np.concatenate((np.array([df['latitude']],dtype=np.float32),np.array([df['longitude']],dtype=np.float32))))
    # # core_samples,cluster_labels=dbscan(X,eps=0.4,metric=distance,min_samples=3)
    # # print('core samples:{}'.format(core_samples))
    # counts_array = np.load('counts_array.npy')
    # df = pd.read_csv(path)
    # # X,node_which_cluster = subsample(counts_array,list(df_only['latitude']),list(df_only['longitude']),df,5000)
    # eps = 0.606
    # min_samples = 500
    # X = np.transpose(np.concatenate((np.array([df_only['latitude']]),np.array([df_only['longitude']]))))
    # core_samples,labels = dbscan(X,eps=eps,min_samples=min_samples,metric=distance,sample_weight=counts_array.astype(np.int64))
    # # core_samples,labels = dbscan(X,eps=eps,min_samples=min_samples,metric=distance,sample_weight=None)
    # # TODO:1. 计算簇数
    # '''
    #     簇数 = max(labels)
    # '''
    # cluster_num = max(labels)+1
    # print('Final Cluster Num:{}'.format(cluster_num))
    # # TODO:2. 计算recall
    # '''
    #     第一件事是总订单数
    #     labels 到 counts_array 的映射直接计算(counts_array[labels!=-1]/np.sum(counts_array))
    # '''
    
    # # recall,dict_for_lat2label=calculate_recall(counts_array=counts_array,node_which_cluster=node_which_cluster,labels=labels,lat_list=list(df_only['latitude']))
    # recall = np.sum(counts_array[np.nonzero(labels!=-1)])/np.sum(counts_array)
    # print('Recall:{}'.format(recall))
    # dict_for_lng2label = dict(zip(list(df_only['longitude']),labels.tolist()))
    # map = np.load('processed_data/processed_re_labeled_map.npy')
    # map = np.flipud(map)
    # output_result = [[] for i in range(max(labels)+1)]
    # lat_min = 41.643
    # lat_max = 42.03
    # lng_min = -87.941
    # lng_max = -87.52
    # node_which_cluster = [-1 for i in range(161)]
    # for lat,lng,label in zip(df_only['latitude'],df_only['longitude'],labels):
    #     if lat>lat_min and lat<lat_max and lng>lng_min and lng<lng_max:
    #         if map[int((lat-lat_min)//0.0001),int((lng-lng_min)//0.0001)] == 0 or label == -1:
    #             continue
    #         node_which_cluster[map[int((lat-lat_min)//0.0001),int((lng-lng_min)//0.0001)]-1] = label
    # output_result = [[] for i in range(max(node_which_cluster)+1)]
    # for ind,component in enumerate(node_which_cluster):
    #     if component != -1:
    #         output_result[component].append(ind)
    # for i in range(len(output_result)-1,-1,-1):
    #     if len(output_result[i])==0:
    #         del output_result[i]
    target_cluster = [
    3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 79, 83, 84, 86, 87, 88,
    89, 90, 125, 129, 130, 131, 132, 133, 140, 141, 142, 143, 144, 147, 148,
    149, 150, 151, 152, 153, 154, 155, 156, 158
]
    eps = 0.71
    min_samples = 212
    import numpy as np
    demand_series = np.load('processed_data/processed_stmatrix.npy')
    center_list = []
    weight_list = []
    for element in target_cluster:
        latlng_array = np.loadtxt('processed_data/Geocoords/'+str(element+1)+'.txt')
        center_list.append(np.mean(latlng_array,axis=0))
        weight_list.append(np.sum(demand_series[:,element]))
    X =np.array(center_list)
    core_samples,labels = dbscan(X,eps=eps,min_samples=min_samples,metric=distance,sample_weight=np.array(weight_list).astype(np.int64))
    output_result = [[] for i in range(max(labels)+1)]
    for index,label in enumerate(labels):
        output_result[label].append(target_cluster[index])
    print(len(output_result))
    with open('dbscan_cluster_1.pkl','wb') as fp:
        pkl.dump(output_result,fp)
    
    # demand_series,node_station_info = make_stmatrix_and_node_station_info(df,dict_for_lng2label,cluster_num)
    # build_uctb_dataset(traffic_node=demand_series,node_station_info=node_station_info,time_fitness=15,time_range=['2018-10-01','2019-05-01'],dataset_name='dbscan_taxi_demand',city='Chicago')
    # # # # TODO:3. 计算自相关性并返回时空矩阵
    # # # '''
    # # #     分配一个时空矩阵
    # # #     对每一个簇(labels里存储着相关信息)，得到一个latlnglist
    # # #     遍历latlnglist找到df着对应的所有record，
    # # # '''
    # print('average acf:{}'.format(avgACF(demand_series, [[node] for node in range(cluster_num)])))
