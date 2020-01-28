from preprocess_data import *
import cProfile
import pstats
import subprocess
from model import DFM
from sklearn.metrics import log_loss

def train_dfm(X, positions, y, timestamps, num_features, args={}, debug=False):
    model = DFM(use_score_cache=True)

    if debug:
        profiler = cProfile.Profile()
        profiler.runcall(model.fit,
                                   X=X,
                                   X_positions=positions,
                                   num_features=num_features,
                                   y=y,
                                   timestamps=timestamps,
                                   **args)
        profiler.dump_stats("./delayed_feedback.stats")
        stats = pstats.Stats("./delayed_feedback.stats")
        stats.sort_stats('time').print_stats(5)
    else:
        model.fit(
            X=X,
            X_positions=positions,
            num_features=num_features,
            y=y, 
            timestamps=timestamps, **args)

    return model

def main(model_name, debug=False)
    print("Reading Data")
    data = read_raw_crite_data()
    hashed_data = make_features_for_cvr_prediction(data)
    
    print("Finish Reading Data")
    if debug:
        print("Debug True")
        hashed_data = hashed_data[::10000]
        ts_click = ts_click[::10000]
        ts_cv = ts_cv[::10000]

    for day in range(54, 61):
        print(f"Start Learning Day{day}")
        train_X, train_positions, train_y, test_X, test_positions, test_y, timestamps = create_data(day, hashed_data, ts_click, ts_cv)

        dfm = train_dfm(train_X, train_positions, train_y, timestamps, 2**24, {}, debug)
        scr = model.score(test_X, test_positions, test_y)
        print(f"Finish Learning Day{day}")
        print(f"Score Day{day}: {scr}")

    return model

if __name__=='__main__':
    import ipdb; ipdb.set_trace()
    dfm_model = main("dfm", debug=False)
