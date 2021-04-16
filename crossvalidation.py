from model_metrics import *
from helpers import *

import turicreate as tc
from datetime import datetime
from itertools import combinations

SEED = 42

def crossval_classic(data, folds, model_name):
    print("K-FOLD CROSS VALIDATION")
    print(f"{folds} folds")
    print(f"start time: {datetime.now().strftime('%H:%M:%S')}")

    models = []

    test_size = int(len(data) / folds)
    data_shuffled = data.copy().shuffle()

    for k in range(0, len(data_shuffled), test_size):
        test_data = data_shuffled[k:k + test_size]
        train_data = data_shuffled.filter_by(filter(lambda path: path not in test_data["path"], data_shuffled["path"]),
                                             "path")

        classifier = tc.image_classifier.create(train_data,
                                                target="label",
                                                model=model_name,
                                                validation_set=None,
                                                seed=SEED,
                                                verbose=False)

        evaluation = classifier.evaluate(test_data, verbose=False)
        metrics = Metrics(train_size=len(train_data), test_size=len(test_data), metrics=evaluation)
        model = Model(classifier, metrics)

        models.append(model)

        print_progress(int(k / test_size), int(len(data_shuffled) / test_size))

    print(f"end time: {datetime.now().strftime('%H:%M:%S')}\n")
    return models


def crossval_location(data, locs, n_train_locs, train_size, test_size, model_name):
    print("LOCATION-BASED CROSS VALIDATION")
    print(f"{n_train_locs} training locations")
    print(f"start time: {datetime.now().strftime('%H:%M:%S')}")

    models = []
    location_combinations = list(combinations(locs, n_train_locs))

    for combination in location_combinations:
        train_locs = list(combination)
        test_locs = list(set(locs) - set(combination))

        train_data = build_sframe_with_locs(data, train_locs, train_size)
        test_data = build_sframe_with_locs(data, test_locs, test_size)

        classifier = tc.image_classifier.create(train_data,
                                                target="label",
                                                model=model_name,
                                                validation_set=None,
                                                seed=SEED,
                                                verbose=False)
        evaluation = classifier.evaluate(test_data, verbose=False)

        metrics = Metrics(train_locs, test_locs, len(train_data), len(test_data), evaluation)
        model = Model(classifier, metrics)

        models.append(model)

        print_progress(location_combinations.index(combination), len(location_combinations))

    print(f"end time: {datetime.now().strftime('%H:%M:%S')}\n")
    return models
