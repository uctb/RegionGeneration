from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np
from shapely.geometry import Polygon, Point


def spatial_aggregation(
        uctb_data: Dict, 
        regions: List[Polygon], 
        regions_build_time: List[str] = None,
        regions_name: List[str] = None,
        agg_func: Callable = np.sum
    ):
    """Aggregates the data from the UCTB data to the regions specified in the regions dictionary.

    In the station info of the returend UCTBData object, the id will be reset;
    the build-time will be the most recent build-time in the region, if regions_build_time is not provided.
    the name will be keep same as the id, if regions_name is not provided.
    
    Args:
        uctb_data: UCTBData object
        regions: list with each element as shapely.geometry.Polygon
        regions_build_time: list of strings, the date format should be '%m/%d/%Y %H:%M'
        regions_name: list of strings, the name of the regions
        agg_func: the function to aggregate the traffic data, default is np.sum
    
    Returns:
        new_uctb_data: UCTBData object after spatial aggregation
    """
    station_info = uctb_data['Node']['StationInfo']
    node_traffic = uctb_data['Node']['TrafficNode']

    # element as [id, [build-time], lat, lng, name]
    new_station_info = [[None, [], r.centroid.y, r.centroid.x, ''] for r in regions]
    station_map = {}

    for s_idx, station in enumerate(station_info):
        for r_idx, region in enumerate(regions):
            if region.contains(Point(station[3], station[2])):
                # id
                new_station_info[r_idx][0] = -1  
                # build_time
                if regions_build_time is None:
                    new_station_info[r_idx][1].append(station[1])  
                else:
                    new_station_info[r_idx][1] = regions_build_time[r_idx]
                # name
                if regions_name is not None:
                    new_station_info[r_idx][4] = regions_name[r_idx]

                station_map[s_idx] = r_idx
                break

    # some regions may not include any station
    new_station_info = [s for s in new_station_info if s[0] is not None]

    # some stations may not be included in any region
    r_idx = len(regions)
    for s_idx, station in enumerate(station_info):
        if station_map.get(s_idx) is None:
            station_map[s_idx] = r_idx
            new_station_info.append(station)
            r_idx += 1

    # aggregate the traffic data
    new_node_traffic = aggregate_node_traffic(node_traffic, station_map, agg_func)

    # reset the id of the stations to 1, 2, 3, ...
    for i, station in enumerate(new_station_info):
        station[0] = i + 1
        if regions_name is None:
            station[4] = str(i + 1)

    # fix the recent build time
    date_format = '%m/%d/%Y %H:%M'
    if regions_build_time is None:
        for station in new_station_info:
            if not isinstance(station[1], list):
                continue
            all_build_time = [datetime.strptime(t, date_format).timestamp() for t in station[1]]
            station[1] = datetime.fromtimestamp(max(all_build_time)).strftime(date_format)


    new_uctb_data = deepcopy(uctb_data)
    new_uctb_data['Node']['TrafficNode'] = new_node_traffic
    new_uctb_data['Node']['StationInfo'] = new_station_info
    
    return new_uctb_data


def aggregate_node_traffic(
    node_traffic: np.array, 
    station_map: Dict[int, int],
    func: Callable = np.sum
    ) -> np.array:
    node_traffic = node_traffic.T  # convert to shape of (num-of-node, time_slots)
    
    clusters = defaultdict(list)
    for s_idx, r_idx in station_map.items():
        clusters[r_idx].append(node_traffic[s_idx])

    r_idx_max = max(station_map.values())
    new_node_traffic = [func(clusters[i], axis=0) for i in range(r_idx_max + 1) if clusters[i] != []]
    new_node_traffic = np.array(new_node_traffic)

    return new_node_traffic.T


