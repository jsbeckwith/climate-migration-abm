from itertools import combinations
import pandas as pd
import math as m
import pickle

def create_distance_dict():
    county_pairs = combinations(range(74), 2)
    coordinates = pd.read_csv('coordinates.csv')
    coord_list = coordinates.to_dict('list')
    # distance_dict = {}
    accurate_distance_dict = {}
    """
    for pair in county_pairs:
        x1 = coord_list['x'][pair[0]]
        y1 = coord_list['y'][pair[0]]
        x2 = coord_list['x'][pair[1]]
        y2 = coord_list['y'][pair[1]]
        distance = m.sqrt((x1-x2)**2 + (y1-y2)**2)
        distance_dict[pair] = distance
    """
    # x is long, y is lat
    for pair in county_pairs:
        x1 = coord_list['x'][pair[0]]
        y1 = coord_list['y'][pair[0]]
        x2 = coord_list['x'][pair[1]]
        y2 = coord_list['y'][pair[1]]
        dlon = m.radians(x2 - x1)
        dlat = m.radians(y2 - y1) 
        a = (m.sin(dlat/2))**2 + m.cos(m.radians(y1)) * m.cos(m.radians(y2)) * (m.sin(dlon/2))**2 
        c = 2 * m.atan2( m.sqrt(a), m.sqrt(1-a) ) 
        d = 3961 * c
        if pair == (9, 30):
            print(d)
        accurate_distance_dict[pair] = int(d)

    with open('distance_dict.pickle', 'wb') as f:
        pickle.dump(accurate_distance_dict, f, pickle.DEFAULT_PROTOCOL)

def create_edge_migration():
    county_pairs = combinations(range(74), 2)
    migration_dict = {}
    for pair in county_pairs:
        migration_dict[pair] = 0
    
    with open('migration_pair_dict.pickle', 'wb') as f:
        pickle.dump(migration_dict, f, pickle.DEFAULT_PROTOCOL)

create_distance_dict()