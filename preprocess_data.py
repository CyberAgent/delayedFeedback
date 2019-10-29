import sys
import pandas as pd
import numpy as np
import pickle
import gc
from const import *
from itertools import combinations, chain
from scipy.sparse import csr_matrix
import scipy.sparse as sp


def read_raw_data():
    '''
    read the data
    :return: pd.DataFrame
    '''
    header = ['ts_click', 'ts_cv', 'int1', 'int2', 'int3', 'int4',
              'int5', 'int6', 'int7', 'int8', 'cat1', 'cat2', 'cat3',
              'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']
    raw_data = pd.read_table(SETTING['data_path'], names=header)
    return raw_data


def save_in_pickle_form(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def check_cv_is_observed(ts_cv, ts_beginning_test_for_cvr):
    '''
    check if cv of the data is observed. If not, ts_cv is made NaN
    :param ts_cv: np.array
    :param ts_beginning_test_for_cvr: int
    '''
    ts_cv[ts_cv > ts_beginning_test_for_cvr] = np.nan


def drop_ts_cols(data):
    '''
    drop the timestamps columns
    :param data: pd.DataFrame
    :return: None
    '''
    data.drop(['ts_click', 'ts_cv'], axis=1, inplace=True)


def bucketize_int_features(data):
    '''
    The integer columns in the data change to categorical ones
    :param data: pd.DataFrame
    '''
    ALT_NA = 999999999
    for feature in data.columns:
        if (data[feature].dtype == np.float64 or data[feature].dtype == np.int64):
            col = data.loc[:, feature]
            col = col.fillna(ALT_NA).astype(np.int64)
            col = np.array(col)
            col = np.where((col > 0) & (col != ALT_NA), (np.log(col + 0.5) / np.log(1.5)).astype(int), col)
            col = col.astype(np.float64)
            col[col == ALT_NA] = np.nan
            data.loc[:, feature] = col


def get_cross_features(row):
    '''
    the row in the data is converted to one with cross features
    :param row:
    :return: np.array
    '''
    index_nonNA = row.notnull()
    row = np.array(row, dtype='str').astype("object")
    i = np.arange(row.size)
    i = i[index_nonNA]
    row = row[index_nonNA]
    features = i.astype('str').astype("object") + '_' + row
    iter_cross_feat = combinations(features, 2)
    return np.concatenate([features, [''.join(x) for x in iter_cross_feat]])


def make_features_for_cvr_prediction(data):
    '''
    create the features matrix in csr_matrix form
    :param data: pd.DataFrame
    :return: np.array
    '''
    bucketize_int_features(data)
    hashed_feature_matrix = [' '.join([str(hashing_trick_py(str(f))) for f in get_cross_features(r)]) for i, r in data.iterrows()]
    return np.array(hashed_feature_matrix).astype('O')

def to_csr_matrix(hashed_feature):
    row = np.repeat(range(len(hashed_feature)), [len(r.split(" ")) for r in hashed_feature])
    col = np.array(" ".join(hashed_feature).split(" "), dtype=int)
    data = [1] * len(col)
    return csr_matrix((data, (row, col)))

# hashing trick
if (sys.version_info > (3, 0)):
    def xrange(a, b, c):
        return range(a, b, c)


    def xencode(x):
        if isinstance(x, bytes) or isinstance(x, bytearray):
            return x
        else:
            return x.encode()
else:
    def xencode(x):
        return x


def hash_string(key, seed=0x0):
    '''
    Implements 32bit murmur3 hash.  Compatible with Scala's MurmurHash3.StringHash
    See:
        - https://github.com/scala/scala/blob/v2.12.3/src/library/scala/util/hashing/MurmurHash3.scala#L72
    '''

    key = bytearray(xencode(key))

    def fmix(h):
        h ^= h >> 16
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h ^= h >> 16
        return h

    length = len(key)
    nblocks = int(length / 2)

    h1 = seed

    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    # body
    for block_start in xrange(0, nblocks * 2, 2):
        # ??? big endian?
        k1 = (key[block_start + 0] << 16) + \
             key[block_start + 1]

        k1 = (c1 * k1) & 0xFFFFFFFF
        k1 = (k1 << 15 | k1 >> 17) & 0xFFFFFFFF  # inlined ROTL32
        k1 = (c2 * k1) & 0xFFFFFFFF

        h1 ^= k1
        h1 = (h1 << 13 | h1 >> 19) & 0xFFFFFFFF  # inlined ROTL32
        h1 = (h1 * 5 + 0xe6546b64) & 0xFFFFFFFF

    # tail
    tail_index = nblocks * 2
    k1 = 0
    tail_size = length & 1

    if tail_size >= 1:
        k1 ^= key[tail_index + 0]

    if tail_size > 0:
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = (k1 << 15 | k1 >> 17) & 0xFFFFFFFF  # inlined ROTL32
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    # finalization
    unsigned_val = fmix(h1 ^ length)
    if unsigned_val & 0x80000000 == 0:
        return unsigned_val
    else:
        return -((unsigned_val ^ 0xFFFFFFFF) + 1)


def hashing_trick_py(str):
    '''
    hash string
    :param str:
    :return int:
    '''
    seed = 1
    vector_length = 2 ** 24
    int_max_scala = 2147483647

    value = hash_string(str, seed)

    if value < 0:
        return (value + int_max_scala) % vector_length
    else:
        return value % vector_length


def save_hashed_feature(hash_data):
    path = SETTING['hashed_data_path_1']
    save_in_pickle_form(hash_data[:int(hash_data.shape[0]/2)], path)
    path = SETTING['hashed_data_path_2']
    save_in_pickle_form(hash_data[int(hash_data.shape[0]/2):], path)


def load_hashed_feature():
    path = SETTING['hashed_data_path_1']
    hashed_feature_1 = load_pickle(path)
    path = SETTING['hashed_data_path_2']
    hashed_feature_2 = load_pickle(path)
    return np.concatenate((hashed_feature_1, hashed_feature_2))


def save_train_data_for_vowpal_wabbit(feature, is_cv, weight, day):
    print("create vowpal wabbit data")
    data_for_cvr_prediction = np.where(is_cv, '1', '-1').astype('O') + ' ' \
            + weight.astype(str).astype('O') + \
            ' | ' + feature + '\n'
    print("DONE")
    print("Join the data")
    data_for_cvr_prediction = ' '.join([x for x in data_for_cvr_prediction])
    print("DONE")

    path = SETTING['train_output_path']
    print("Saving the data: " + str(path))
    with open(path + 'train_for_cvr' + str(day) + '.txt', "w") as f:
        f.write(data_for_cvr_prediction)
    print("DONE")


def save_test_data_for_vowpal_wabbit(feature, is_cv, day):
    print("create vowpal wabbit data")
    test_data = np.where(is_cv, '1', '-1').astype('O') + ' | ' + feature +'\n'
    print("Done")
    print("saving the data")
    path = SETTING['test_output_path']
    with open(path + 'test_for_cvr' + str(day) + '.txt', "w") as f:
        f.write(' '.join([x for x in test_data]))
    print("done")


if __name__=='__main__':
    # preprocessing for CVR prediction
    print('read  the data')
    data = read_raw_data()
    ts_start = SECONDS_PER_DAY * 32
    ts_end   = SECONDS_PER_DAY * 60
    data = data[(data.ts_click >= ts_start) & (data.ts_click < ts_end)]
    ts_click = data.ts_click
    ts_cv = data.ts_cv
    drop_ts_cols(data)
    hashed_feature = make_features_for_cvr_prediction(data)
    save_hashed_feature(hashed_feature)

