"""
Load the raw data.
"""
import json
import re 
import pandas as pd

from os import listdir


def get_json_basenames(basenames: list[str]) -> list[str]:
    """
    Return the elements of basenames that end with '.json'.

    :param basenames: A list of basenames
    :type basenames: list[str]

    :return: The elements of basenames that end with '.json'
    :rtype: list[str]
    """
    json_basenames = []
    for basename in basenames:
        if re.fullmatch(r'.*\.json', basename) is not None:
            json_basenames.append(basename)
    return json_basenames


def load_jsons(json_basenames: list[str], dirname: str) -> list:
    """
    Return a list of json objects.

    :param json_basenames: A list of basenames each ending with '.json'
    :type json_basename: list[str]

    :param dirname: The name of the directory containing the basenames
    :type dirname: str

    :return: A list of json objects, each loaded from a basename
    :rtype: list
    """
    jsons = []
    for basename in json_basenames:
        filename = dirname + basename
        try:
            with open(filename, 'r') as f:
                json_obj = json.load(f)
        except FileNotFoundError:
            print(f'FileNotFoundError: {filename} does not exist')
        except json.JSONDecodeError:
            print(f'JSONDecodeError: {filename} does not contain valid JSON data')
        else:
            jsons.append(json_obj)
    return jsons


def filter_house_properties(city: dict) -> list[dict]:
    """
    Filter the potentially relevant features from city.
    """
    try:
        houses: list[dict] = city['data']['results'] 
    except KeyError:
        print(f'KeyError: there are no houses')
        return None
    
    # a list of house properties worth keeping
    keepers = ['property_id', 'status', 'list_date', 
               'location', 'description', 'tags',
               'flags', 'community', 'open_houses']
    
    house_data = []
    for house in houses:
        house_props = {k: house.get(k) for k in keepers}
        house_data.append(house_props)

    return house_data


def parse_location(house: dict) -> dict:
    """
    Return a house dictionary with 'location' parsed.
    """
    if 'location' not in house.keys():
        return house
    
    house_copy: dict = house.copy()
    location: dict = house_copy.pop('location')

    if 'address' not in location.keys():
        return house

    address: dict = location['address']   
    loc_data = {
        'postal_code': address.get('postal_code'),
        'state': address.get('state'),
        'city': address.get('city')
    }

    coordinate = address.get('coordinate')
    if coordinate is not None:
            loc_data['lon'] = coordinate.get('lon')
            loc_data['lat'] = coordinate.get('lat')

    return {**house_copy, **loc_data}


def flatten_description(house: dict) -> dict:
    """
    Return a house dictionary with 'description' flattened.
    """
    if 'description' not in house.keys():
        return house

    house_copy: dict = house.copy()
    description: dict = house_copy.pop('description')
    return {**house_copy, **description}


def parse_tags(house: dict) -> dict:
    """
    Return a house dictionary with 'tags' parsed.
    """
    if 'tags' not in house.keys():
        return house
    
    house_copy: dict = house.copy()
    tags: list = house_copy.pop('tags')
    if tags is None:
        return house
    
    tags_dict = {tag: True for tag in tags}
    return {**house_copy, **tags_dict}


def load_data(dirname) -> pd.DataFrame:
    """
    Meh.
    """ 
    basenames: list[str] = listdir(path=dirname)
    json_basenames: list[str] = get_json_basenames(basenames)    
    cities: list[dict] = load_jsons(json_basenames, dirname)
    
    all_house_data = []
    for city in cities:
        city_house_data = filter_house_properties(city)
        all_house_data.extend(city_house_data)

    all_house_data =\
        [flatten_description(house) for house in all_house_data]
    
    all_house_data =\
        [parse_location(house) for house in all_house_data]
    
    all_house_data =\
        [parse_tags(house) for house in all_house_data]

    return all_house_data
    

if __name__ == '__main__':
    # print(__doc__)
    data: list = load_data('data/raw/')
    print(f'type: {type(data)}')
    print(f'length: {len(data)}')
    idx = 1000
    print()
    print(data[idx])
 
