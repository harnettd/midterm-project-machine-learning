"""
Load raw housing data from a collection of JSON files.
Filter out uninteresting properties from the housing data.
Flatten nested lists and dictionaries within the housing data.
Create a pandas DataFrame from the housing data.
Export the DataFrame to a CSV.
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


def load_jsons(json_basenames: list[str], dirname: str) -> list[dict]:
    """
    Return a list of loaded json objects (dictionaries).

    :param json_basenames: A list of basenames each ending with '.json'
    :type json_basename: list[str]

    :param dirname: The name of the directory containing the basenames
    :type dirname: str

    :return: A list of json objects, each loaded from a basename
    :rtype: list[dict]
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


def filter_house_properties(city: dict) -> list[dict] | None:
    """
    Filter the potentially relevant house features from city.

    Keep information on listing ID, status, list date, location,
    description, tags, flags, community, and open_houses.

    :param city: A particular city (or part thereof)
    :type city: dict

    :return: A list of houses in the city
    :rtype: list[dict]
    """
    try:
        houses: list[dict] = city['data']['results'] 
    except KeyError:
        return None
    
    # a list of house properties worth keeping
    keepers = ['listing_id', 'status', 'list_date', 
               'location', 'description', 'tags',
               'flags', 'community', 'open_houses']
    
    house_data = []
    for house in houses:
        house_props = {k: house.get(k) for k in keepers}
        house_data.append(house_props)

    return house_data


def flatten(house: dict, key: str) -> dict:
    """
    Return a house with nested dictionary corresponding to key flattened.

    :param house: A particular house
    :type house: dict

    :param key: The key of a nested dictionary in house
    :type key: str

    :return: The house with a flattened (previously) nested dictionary
    :rtype: dict
    """
    house_copy: dict = house.copy()
    try:
        key_dict: dict = house_copy.pop(key)
    except KeyError:
        return house
    else:
        return {**house_copy, **key_dict}
    

def parse_tags(house: dict) -> dict:
    """
    Return a house (dictionary) with tags parsed.

    Keep all tags.

    :param house: A particular house
    :type house: dict

    :return: The house, but with information extracted from tags
    :rtype: dict
    """    
    house_copy: dict = house.copy()
    try:
        tags: list = house_copy.pop('tags')
    except KeyError:
        return house 
    else:
        if tags is None:
            tags = []
        tags_dict = {tag: True for tag in tags}
    
    return {**house_copy, **tags_dict}


def parse_location(house: dict) -> dict:
    """
    Return a house (dictionary) with location parsed.

    Keep information on city, state, postal code, latitude, 
    and longitude when available.

    :param house: A particular house
    :type house: dict

    :return: The house, but with information extracted from tags
    :rtype: dict
    """
    house_copy: dict = house.copy()
    try:
        address: dict = house_copy['location'].pop('address')
    except KeyError:
        return house
    else:   
        loc_data = {
            'postal_code': address.get('postal_code'),
            'state': address.get('state'),
            'city': address.get('city')
        }

        try:
            coordinate = address['coordinate']
        except KeyError:
            pass
        else:
            if coordinate is not None:
                loc_data['lon'] = coordinate['lon']
                loc_data['lat'] = coordinate['lat']

    return {**house_copy, **loc_data}


def load_data(dirname: str) -> pd.DataFrame:
    """
    Load housing data from all JSON files into a DataFrame.

    :param dirname: Data directory name
    :type dirname: str

    :return: A DataFrame of housing data
    :rtype: DataFrame
    """ 
    # Make a list of all JSON basenames in the data directory.
    basenames: list[str] = listdir(path=dirname)
    json_basenames: list[str] = get_json_basenames(basenames)

    # Read in JSON files. Note that each file corresponds to (part of) 
    # a city. Each element of a city corresponds to a house.
    cities: list[dict] = load_jsons(json_basenames, dirname)
    
    # Loop through all cities, filtering out house properties that don't
    # seem relevant to an analysis of sales prices.
    all_house_data: list[dict] = []  # housing data for *all* houses in *all* cities
    for city in cities:
        city_house_data: list[dict] = filter_house_properties(city)
        all_house_data.extend(city_house_data)

    # Before creating a DataFrame, all nested dictionaries and lists
    # need to be flattened.
    all_house_data =\
        [flatten(house, 'description') for house in all_house_data]
    
    all_house_data =\
        [flatten(house, 'flags') for house in all_house_data]
    
    all_house_data =\
        [parse_location(house) for house in all_house_data]
    
    all_house_data =\
        [parse_tags(house) for house in all_house_data]
 
    all_house_df = pd.DataFrame(all_house_data)
    
    return all_house_df
    

if __name__ == '__main__':
    # Load housing data, process it (a bit!), and then export to CSV.
    data_dirname = 'data/'
    data_df = load_data(data_dirname + 'raw/')
    data_df.to_csv(data_dirname + 'processed/housing_data_0.csv', sep=',')
