import numpy as np
from statsmodels.tsa.stattools import acf

grid = np.load('processed_data/grid.npy')
grid_sum = np.sum(grid,axis=0,keepdims=False)
m,n = grid_sum.shape
grid_nonzero_list = np.argwhere(grid_sum>=212)
print(np.sum(grid_sum[np.nonzero(grid_sum>=212)])/np.sum(grid_sum))
print(len(grid_nonzero_list))
m,n = grid_sum.shape
print(m,n)
acf_list = []
for coord in grid_nonzero_list:
    y = coord[0]
    x = coord[1]
    acf_list.append(acf(grid[:,y,x],nlags=96)[95])
print(np.mean(np.array(acf_list)))



