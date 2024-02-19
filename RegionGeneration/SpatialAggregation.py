from shapely.geometry import Polygon,Point
from copy import copy,deepcopy
def spatial_aggregation(uctb_data,regions):
    """
    This function aggregates the data from the UCTB data to the regions specified in the regions dictionary.
    Args:
        uctb_data: UCTBData object
        regions: list with each element as shapely.geometry.Polygon
    Returns:
        new_uctb_data: UCTBData object after spatial aggregation
    """
    clusters = []
    station_info = uctb_data['Node']['StationInfo']
    node_traffic = uctb_data['Node']['TrafficNode']
    # Input station_info,node_traffic output new_station_info,new_node_traffic
    new_station_info = []
    new_node_traffic = []
    new_uctb_data = deepcopy(uctb_data)
    new_uctb_data['Node']['TrafficNode'] = new_node_traffic
    new_uctb_data['Node']['StationInfo'] = new_station_info
    return new_station_info
        
    