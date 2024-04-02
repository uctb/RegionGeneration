import numpy as np
from statsmodels.tsa.stattools import acf
import networkx as nx
from scipy.stats import pearsonr
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
def avgCorr(demand_series,output_result):
    cor_list = []
    for cluster in output_result:
        tmp_demand_series = np.transpose(demand_series[:,cluster]) 
        cor_list.append(np.sum(np.corrcoef(tmp_demand_series)-np.eye(len(cluster))))
    return np.mean(np.array(cor_list))
def avgACF(demand_series,output_result):
    acf_list = []
    for cluster in output_result:
        acf_list.append(calculate_acf(demand_series,cluster,96))
    return np.mean(np.array(acf_list))

def calculate_acf(demand_series,node_list,most_lags):
    if len(node_list) == 0:
        return 0
    tmp_series = np.sum(demand_series[:,node_list],axis=1,keepdims=False)
    try:
        return acf(tmp_series,nlags=most_lags)[-1]
    except:
        return 0

def avgCoverage(order_area_vec,total_area_vec,output_result):
    sum = 0
    for index,cluster in enumerate(output_result):
        sum += np.sum(order_area_vec[cluster])/np.sum(total_area_vec[cluster])
    return sum / index    

def Recall(output_result,lat_list,lng_list,map,lat_min,lat_max,lng_min,lng_max,lat_lateral,lng_lateral):
    assert len(lat_list)==len(lng_list)
    hit = 0
    total = len(lat_list)
    hit_node_list = []
    for cluster in output_result:
        hit_node_list.extend(cluster)
    for lat,lng in zip(lat_list,lng_list):
        if lat >= lat_max or lng >= lng_max:
            continue
        x = int((lat-lat_min)//lat_lateral)
        y = int((lng-lng_min)//lng_lateral)
        label = map[x,y] - 1
        if label in hit_node_list:
            hit += 1
    return hit / total

def mape(y_pred,y_truth,threshold):

    assert y_pred.shape == y_truth.shape
    m,n,p=y_pred.shape
    y_pred = y_pred.reshape([m,n])
    y_truth = y_truth.reshape([m,n])
    error = np.abs(y_pred-y_truth)
    y_truth_abs = np.abs(y_truth)
    return np.mean(np.divide(error[np.nonzero(y_truth_abs>=threshold)],y_truth_abs[np.nonzero(y_truth_abs>=threshold)] ))
def _valuefunction_2(demand_series,node,cluster,graph,locations):
        tmp_sum = []
        neighbor_list = list(nx.neighbors(graph,node))
        max_dis = 0
        for node_ in cluster:
            if node_ in neighbor_list:
                tmp_dis = haversine_dis(locations[node][1],locations[node][0],locations[node_][1],locations[node_][0])
                if max_dis<tmp_dis:
                    max_dis = tmp_dis
                tmp_sum.append((1+pearsonr(demand_series[:,node],demand_series[:,node_])[0])/2)
        if len(tmp_sum) == 0:
            return -np.inf
        return np.sum(np.array(tmp_sum))#*np.log(self.tau/max_dis)

if __name__ == '__main__':
    node = 56
    cluster_after_add_1 = [57, 144, 44, 56]
    cluster_after_add_2 = [33, 155, 36, 39, 130, 32, 35, 34]
    cluster_after_add_3 = [46, 54, 43, 49, 45, 48, 55]
    demand_series = np.load('processed_data/processed_stmatrix.npy')
    am= np.load('processed_data/processed_Chicago_am.npy')
    area_array = np.load('processed_data/processed_area_info.npy')
    locations = []
    for i in range(1,am.shape[0]+1):
        X = np.loadtxt('processed_data/Geocoords/'+str(i)+'.txt')
        locations.append(np.mean(X,axis=0))
    graph = nx.Graph(am)
    cluster_before_throw = [46,54,43,49,45,48,55,56]
    cluster_after_throw = [46,54,43,49,45,48,55]
    cluster_before_add = [57, 144, 44]
    cluster_after_add = [57, 144, 44, 56]
    print(_valuefunction_2(demand_series,node,cluster_after_add,graph,locations))
    print(_valuefunction_2(demand_series,node,cluster_after_add_1,graph,locations))
    print(_valuefunction_2(demand_series,node,cluster_after_add_2,graph,locations))
    print(_valuefunction_2(demand_series,node,cluster_after_add_3,graph,locations))

    corr_before_throw = avgCorr(demand_series=demand_series,output_result=[cluster_before_throw])
    corr_after_throw = avgCorr(demand_series=demand_series,output_result=[cluster_after_throw])
    corr_before_add = avgCorr(demand_series=demand_series,output_result=[cluster_before_add])
    corr_after_add = avgCorr(demand_series=demand_series,output_result=[cluster_after_add])
    print(corr_before_throw,corr_after_throw)
    print(corr_before_add,corr_after_add)
