# copyright 2010 Eric Gradman
# free to use for any purpose, with or without attribution
# from an algorithm by James McNeill at
# http://playtechs.blogspot.com/2007/04/hex-grids.html

# the center of hex (0,0) is located at cartesian coordinates (0,0)

from datetime import timedelta
from dateutil.parser import parse
import math
import numpy as np
import pandas as pd
# R ~ center of hex to edge
# S ~ edge length, also center to vertex
# T ~ "height of triangle"
class HexagonGridGenerator():
    def __init__(self,real_R) -> None:
        self.real_R = real_R # in my application, a hex is 2*75 pixels wide
        self.R = 2.
        self.S = 2.*self.R/np.sqrt(3.)
        self.T = self.S/2.
        self.SCALE = real_R/self.R

# XM*X = I
# XM = Xinv
        self.X = np.array([
            [ 0, self.R],
            [-self.S, self.S/2.]
        ])
        self.XM = np.array([
            [1./(2.*self.R),  -1./self.S],
            [1./self.R,        0.  ]
        ])
# YM*Y = I
# YM = Yinv
        self.Y = np.array([
            [self.R,    -self.R],
            [self.S/2.,  self.S/2.]
        ])
        self.YM = np.array([
            [ 1./(2.*self.R), 1./self.S],
            [-1./(2.*self.R), 1./self.S],
        ])

        pass

    def cartesian2hex(self,cp):
        """convert cartesian point cp to hex coord hp"""
        cp = np.multiply(cp, 1./self.SCALE)
        Mi = np.floor(np.dot(self.XM, cp))
        xi, yi = Mi
        i = np.floor((xi+yi+2.)/3.)

        Mj = np.floor(np.dot(self.YM, cp))
        xj, yj = Mj
        j = np.floor((xj+yj+2.)/3.)

        hp = i,j
        return hp

    def hex2cartesian(self,hp):
        """convert hex center coordinate hp to cartesian centerpoint cp"""
        i,j = hp
        cp = np.array([
            i*(2*self.R) + j*self.R,
            j*(self.S+self.T)
        ])
        cp = np.multiply(cp, self.SCALE)
        return cp

def get_hexagon_latlng_list(xnum,origin_lat,origin_lng,index,scale):
    x = index%xnum
    y = index//xnum
    constant = real_R/scale
    lng = origin_lng+2*x*constant+2*y*constant*math.cos(math.pi/3)
    lat = origin_lat-2*y*constant*math.sin(math.pi/3)
    lat_1 = lat + constant/np.sqrt(3)*2
    lng_1 = lng
    lat_2 = lat + constant/np.sqrt(3)
    lng_2 = lng + constant
    lat_3 = lat - constant/np.sqrt(3)
    lng_3 = lng + constant
    lat_4 = lat - constant/np.sqrt(3)*2
    lng_4 = lng
    lat_5 = lat - constant/np.sqrt(3)
    lng_5 = lng - constant
    lat_6 = lat + constant/np.sqrt(3)
    lng_6 = lng - constant

    return [[lat_1,lng_1],[lat_2,lng_2],[lat_3,lng_3],[lat_4,lng_4],[lat_5,lng_5],[lat_6,lng_6]]


if __name__ == '__main__':
    origin_lat = 42.05
    origin_lng = -88.09
    scale = 1000
    a = 0.49
    # import folium
    # from folium.plugins import HeatMap
    # mymap = folium.Map([41.8,-87.9])
    # real_R = 10
    # mymap.add_child(folium.LatLngPopup())
    # print(cartesian2hex((245,0)))
    df = pd.read_csv('E:/project/mission/didi/data/event/taxi.csv')
    # latlng_array = np.transpose(np.concatenate((np.array([df['latitude']]),np.array([df['longitude']]))))
    # xnum = int(a/(2*real_R)*scale)+1
    # ynum = xnum
    td = timedelta(minutes=15)
    # for i in range(0,xnum*ynum):
    #     folium.Polygon(get_hexagon_latlng_list(xnum,origin_lat,origin_lng,i,scale),popup=str(i),fill=True).add_to(mymap)
    
    reference_date = parse('10/01/2018 12:00:00 AM')
    from statsmodels.tsa.stattools import acf
    for real_R in np.linspace(10.6666666,10.6666666,1):
        hgg = HexagonGridGenerator(real_R)
        xnum = int(a/(2*real_R)*scale)+1
        ynum = xnum
        hexagon_grid = np.zeros([20352,xnum,ynum])
        tmp_latlng_list = []
        for date,lat,lng in zip(df['date'],df['latitude'],df['longitude']):
            tmp_latlng_list.append([lat,lng])
            timeindex = int((parse(date) - reference_date)//td)
            # folium.Circle([lat,lng],radius=10).add_to(mymap)
            x = (lng-origin_lng)*scale
            y = (origin_lat - lat)*scale
            coord = hgg.cartesian2hex((x,y))
        # folium.Polygon(get_hexagon_latlng_list(xnum,origin_lat,origin_lng,int(coord[0]+coord[1]*xnum),scale),color='red',fill=True).add_to(mymap)
            hexagon_grid[timeindex,int(coord[0]),int(coord[1])] += 1
            # HeatMap(tmp_latlng_list).add_to(mymap)
            # print(xnum*coord[1]+coord[0])
    # polygonlist[int(coord[0] + coord[1] * 20)] = folium.Polygon(get_hexagon_latlng_list(20,origin_lat,origin_lng,int(coord[0] + coord[1] * 20),scale),color = 'green')
    # for i in range(0,40):
    #     polygonlist[i].add_to(mymap)
    
        hexagon_grid_sum = np.sum(hexagon_grid,axis = 0,keepdims=False)
        hexagon_grid_nonzero_list = np.argwhere(hexagon_grid_sum>212)
        acf_list = []
        grid = []
        for coord in hexagon_grid_nonzero_list:
            x = coord[0]
            y = coord[1]
            acf_list.append(acf(hexagon_grid[:,x,y],nlags=96)[-1])
            grid.append(hexagon_grid[:,x,y])
        np.save('processed_data/hexagon.npy',hexagon_grid)
        print('real_R:'+str(real_R)+';'+'Node Number:{};'.format(len(hexagon_grid_nonzero_list))+'acf:{}'.format(np.mean(np.array(acf_list))))
        
# 41.9977, -88.0231