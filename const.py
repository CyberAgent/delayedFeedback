import json


def read_json_setting():
    '''
    read json setting file which is located in ./setting.json
    :return: dict
    '''
    with open('setting.json', 'rb') as json_setting:
        setting = json.load(json_setting)
    return setting


SETTING = read_json_setting()
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 60 * 60
TRAIN_DATA_DAYS = 21  # the length of the training term