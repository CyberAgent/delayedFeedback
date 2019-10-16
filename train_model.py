from preprocess_data import *
from estimate_fsiw import *
from vowval_wabbit import *
import cProfile
import pstats
import subprocess
from dfm import DelayedFeedback
from sklearn.metrics import log_loss


def convert_feature(feature):
    data_list = []
    position_list = [0]
    for x in feature:
        data_list.extend(x)
        position_offset = position_list[-1]
        position_list.append(position_offset + len(x))

    data = np.array(data_list, dtype=np.uint32)
    positions = np.array(position_list, dtype=np.intc)
    return data, positions


def score_lr_fsiw(data, chose_model, debug=False):
    if debug:
        data = data.iloc[::1000] # for debug

    ts_start = SECONDS_PER_DAY * 32
    ts_end   = SECONDS_PER_DAY * 60
    used_index = (data.ts_click >= ts_start) & (data.ts_click < ts_end)
    ts_click = data.ts_click.values[used_index]
    ts_cv = data.ts_cv.values[used_index]
    campaign_ids = data.cat3.values[used_index]
    hashed_feature = load_hashed_feature()
    if debug:
        hashed_feature = hashed_feature[::1000]
    scrs = []
    total_nll = 0
    num_data = 0
    for day in range(54, 61):
        ts_beginning_test_for_cvr   = SECONDS_PER_DAY * (day-1)
        ts_starting_train_for_cvr   = ts_beginning_test_for_cvr - 21 * SECONDS_PER_DAY
        ts_end_test_for_cvr         = ts_beginning_test_for_cvr + 1 * SECONDS_PER_DAY

        tmp_ts_click = ts_click[(ts_click >= ts_starting_train_for_cvr) 
                & (ts_click < ts_end_test_for_cvr)]
        tmp_ts_cv    = ts_cv[(ts_click >= ts_starting_train_for_cvr) & (ts_click < ts_end_test_for_cvr)]
        tmp_hashed_feature = hashed_feature[(ts_click >= ts_starting_train_for_cvr) & (ts_click < ts_end_test_for_cvr)]
        is_train = tmp_ts_click < ts_beginning_test_for_cvr
        is_test = ~is_train
        check_cv_is_observed(tmp_ts_cv[is_train], ts_beginning_test_for_cvr)
        is_cv = ~np.isnan(tmp_ts_cv)

        if chose_model == 'fsiw':
            start = time()
            weight = cal_fsiw(day, data)
            end = time()
            print(f"The elapsed time to estimate FSIW {end - start} seconds")

            # training with vowpal wabbit
            save_train_data_for_vowpal_wabbit(tmp_hashed_feature[is_train], is_cv[is_train], weight, day)
            save_test_data_for_vowpal_wabbit(tmp_hashed_feature[is_test], is_cv[is_test], day)
            start = time()
            train_vw(day)
            end = time()
            print(f"The elapsed time to train CVR predictor: {end - start} seconds")
            predict_vw(day)

            prediction = read_pred(day)
            nll_per_day, num_data_per_day = evaluate_cvr_prediction(is_cv[is_test], prediction)
            total_nll += nll_per_day
            num_data += num_data_per_day

            print(f'nll_score: {nll_per_day / num_data_per_day}')
            print('nll_score')
            print(total_nll/num_data)
            camp_ids = campaign_ids[(ts_click >= ts_starting_train_for_cvr) & (ts_click < ts_end_test_for_cvr)][is_test]
            correct_label = is_cv[is_test]
            save_in_pickle_form(camp_ids, 'data/camp_ids_' + str(day))
            save_in_pickle_form(correct_label, 'data/is_cv_' + str(day))
            save_in_pickle_form(prediction, 'data/prediction' + str(day))

        elif chose_model == 'dfm':
            tmp_hashed_feature = list(map(lambda x :list(map(int, x.split())), tmp_hashed_feature))
            model = DelayedFeedback(use_score_cache=True)
            train_timestamps = np.where(is_cv[is_train], tmp_ts_cv[is_train] - tmp_ts_click[is_train],
                    ts_beginning_test_for_cvr - tmp_ts_click[is_train]).astype(int)
            tmp_train_data = np.array(tmp_hashed_feature)[is_train]
            tmp_test_data = np.array(tmp_hashed_feature)[is_test]
            feature_train_data, train_positions = convert_feature(tmp_train_data)
            feature_test_data, test_positions = convert_feature(tmp_test_data)
            start = time()
            if debug:
                profiler = cProfile.Profile()
                results = profiler.runcall(model.fit,
                                           feature_data=feature_train_data,
                                           feature_positions=train_positions,
                                           labels=is_cv[is_train],
                                           timestamps=train_timestamps,
                                           maxiter=1000,
                                           alpha=100)
                profiler.dump_stats("./delayed_feedback.stats")
                stats = pstats.Stats("./delayed_feedback.stats")
                stats.sort_stats('time').print_stats(5)
            else:
                results = model.fit(
                    feature_data=feature_train_data,
                    feature_positions=train_positions,
                    labels=is_cv[is_train],
                    timestamps=train_timestamps, maxiter=100000, alpha=100)
            end = time()

            weights = results['x']
            prediction = model.predict(feature_test_data, test_positions, weights)
            save_in_pickle_form(prediction, 'dfm_prediction' + str(day) + '.pckl')
            nll_per_day = log_loss(is_cv[is_test], prediction, normalize=False)
            num_data_per_day = prediction.shape[0]
            total_nll += nll_per_day
            num_data += num_data_per_day 
            print(f"Day:{day} score is {nll_per_day / num_data_per_day}")
            print(f"The current score is {total_nll / num_data}")
            print(f"The elapsed time training DFM is {end - start} ")
        scrs.append(total_nll/num_data)
        print(f"scores {scrs}")
    return scrs

if __name__=='__main__':
    data = read_raw_data()
    scrs =score_lr_fsiw(data, 'dfm', False)
    import pdb; pdb.set_trace()
