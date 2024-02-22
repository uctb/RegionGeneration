from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import pandas as pd
from shapely.geometry import Polygon, Point
import pandas as pd


def spatial_aggregation(
        uctb_data: Dict, 
        regions: List[Polygon], 
        regions_build_time: List[str] = None,
        regions_name: List[str] = None
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
    
    Returns:
        new_uctb_data: UCTBData object after spatial aggregation
    """
    station_info = uctb_data['Node']['StationInfo']
    if isinstance(station_info[4], list):
        # Bike_Chicago: [..., [name]], instead of [..., name]
        station_info = [[s[0], s[1], s[2], s[3], s[4][0]] for s in station_info]
    node_traffic = uctb_data['Node']['TrafficNode']

    new_station_info = [[None, [], r.centroid.y, r.centroid.x, ''] for r in regions]
    station_map = {}

    for station in station_info:
        for i, region in enumerate(regions):
            if region.contains(Point(station[3], station[2])):
                # id
                station_map[station[0]] = 0
                # build-time
                if regions_build_time is None:
                    new_station_info[i][1].append(station[1])
                else:
                    new_station_info[i][1] = regions_build_time[i]
                # name
                if regions_name is not None:
                    new_station_info[i][4] = regions_name[i]

                station_map[station[4]] = i
                break

    # some regions may not include any station
    new_station_info = [s for s in new_station_info if s[0] is not None]

    # some stations may not be included in any region
    i = len(regions)
    for station in station_info:
        name = station[4]
        if station_map.get(name) is None:
            station_map[name] = i
            new_station_info.append(station)
            i += 1

    # aggregate the traffic data
    df = pd.DataFrame(node_traffic, columns=[s[4] for s in station_info])
    df = df.T

    df['region'] = df.index.map(station_map)
    df = df.groupby('region').sum()
    df = df.T
    new_node_traffic = df.values

    # reset the id of the stations
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