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
    Filter the potentially relevant features from json_obj.
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


def flatten_description(house: dict) -> dict:
    """
    Return a house dictionary with 'description' flattened.
    """
    if 'description' not in house.keys():
        return house

    description = house.pop('description')
    return {**house, **description}


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

    for house in all_house_data:
        house = flatten_description(house)

    return all_house_data
    

if __name__ == '__main__':
    # print(__doc__)
    data: list = load_data('data/raw/')
    print(f'type: {type(data)}')
    print(f'length: {len(data)}')
    # print(data[0])
