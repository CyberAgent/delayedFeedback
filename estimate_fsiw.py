from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from preprocess_data import *
from time import time


def get_timezone(ts, deadline):
    '''
    create a categorical feature indicating timezone from 0 to 23
    :param np.array
    :type deadline int
    '''
    elapse = get_elapse(ts, deadline)
    elapse %= SECONDS_PER_DAY
    elapse = (elapse / SECONDS_PER_HOUR).astype(np.int8)
    return elapse


def get_elapse(timestamps, deadline):
    return np.array(deadline - timestamps)


def make_elapse_feature(timestamps, end, dimensions):
    '''
    create features of elapsed time
    '''
    elapse = get_elapse(timestamps, end).astype(int)
    return np.array([elapse ** i for i in range(1, dimensions + 1)]).T


def add_elapses(data, ts_click, end):
    elapses = make_elapse_feature(ts_click, end, 1)
    data = data.assign(t=elapses)
    return data

def lgb_engineer_features(data, ts_click, cf_deadline):
    '''
    create features to estimate FSIW
    '''
    categorized_ts = get_timezone(ts_click, cf_deadline)
    data = data.assign(time_zone=categorized_ts)
    data.index = range(data.shape[0])
    categories = ['cat' + str(n) for n in range(1, 10)] + ['time_zone']
    for cate in categories:
        data[cate] = LabelEncoder().fit_transform(data[cate].astype(str))

    return data


def create_target_for_fsiw_cv_1(ts_cv, counterfactual_deadline):
    '''
    create the target to estimate fsiw for y=1
    :param ts_cv: np.array
    :param counterfactual_deadline: int
    :return: np.array
    '''
    is_observed = (ts_cv <= counterfactual_deadline)
    return is_observed


def create_traget_for_fsiw_cv_0(ts_cv, counterfactual_deadline):
    '''
    create the target to estimate fsiw for y=0
    :param ts_cv: np.array
    :param counterfactual_deadline: int
    :return: np.array
    '''
    is_observed = create_target_for_fsiw_cv_1(ts_cv, counterfactual_deadline)
    is_conversion = ~np.isnan(ts_cv)
    return ~np.logical_xor(is_conversion, is_observed)


def train_fsiw_with_lgb(data, label, day, label_cv):
    '''
    :param data: pd.DataFrame
    :param label: np.array
    :param day: int
    :return: lgb.Boost
    '''
    params = SETTING['params_' + str(day) + '_' + str(label_cv)]
    print(params)
    dtrain = lgb.Dataset(data=data, label=label)
    clf = lgb.train(train_set=dtrain, params=params)

    return clf


def cal_fsiw(day, data):
    ts_end_train = SECONDS_PER_DAY * (day - 1)
    ts_beginning_train = ts_end_train - 21 * SECONDS_PER_DAY
    counterfactual_deadline = ts_end_train - SETTING['HOW_LONG_CF'] * SECONDS_PER_DAY
    data = data[(data.ts_click >= ts_beginning_train) & (data.ts_click < ts_end_train)]

    ts_click = data.ts_click.values
    ts_cv = data.ts_cv.values
    is_cv = ts_cv <= ts_end_train
    ts_cv[is_cv > ts_end_train] = np.nan
    is_train =  ts_click <= counterfactual_deadline
    drop_ts_cols(data)

    target_1 = ts_cv < counterfactual_deadline
    target_0 = ~np.logical_xor((ts_cv < counterfactual_deadline), is_cv)

    feature = lgb_engineer_features(data, ts_click, counterfactual_deadline)
    feature_1 = add_elapses(feature[is_cv & is_train], ts_click[is_cv & is_train], counterfactual_deadline)
    feature_0 = add_elapses(feature[~(ts_cv < counterfactual_deadline) & is_train], ts_click[~(ts_cv < counterfactual_deadline) & is_train], counterfactual_deadline)
    feature = add_elapses(feature, ts_click, ts_end_train)

    fsiw_1 = train_fsiw_with_lgb(feature_1,  target_1[is_cv & is_train], day, 1).predict(feature)
    fsiw_0 = train_fsiw_with_lgb(feature_0, target_0[~(ts_cv < counterfactual_deadline) & is_train], day, 0).predict(feature)

    return np.where(is_cv, 1/fsiw_1, fsiw_0)
