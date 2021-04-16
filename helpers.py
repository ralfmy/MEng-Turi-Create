import matplotlib.pyplot as plt
import pickle
import os


def build_sframe_with_locs(data, locs, size):
    sframe = data.copy().filter_by(filter(lambda path: any(loc in path for loc in locs), data["path"]), "path")

    if size > len(locs) * 800:
        size = len(locs) * 800

    sframe = sframe.shuffle()[:size]

    return sframe

def get_model_metrics(models):
    return [model.metrics for model in models]

def print_progress(i, total):
    print(f"{i + 1} / {total}", end="\r")


def print_cv_metrics(cv):
    for i in range(len(cv)):
        print(i + 1)
        cv[i].metrics.display()
    print("----------------------------------------\n")


def print_metrics(metrics):
    for i in range(len(metrics)):
        print(i + 1)
        metrics[i].display()
    print("----------------------------------------\n")


def plot_line(x, y, xlabel, ylabel, ymin=0.5, title=""):
    plt.figure()
    plt.plot(x, y)
    plt.xticks(x)
    plt.ylim(ymin, 1.0)
    plt.title(title, pad=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_roc(roc_curve, title=""):
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(roc_curve["fpr"], roc_curve["tpr"])
    plt.title(title)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.show()


# PICKLES
PICKLES_DIR = "./pickles/"


def load_pickle(name):
    try:
        with open(PICKLES_DIR + name + ".pkl", "rb") as p:
            return pickle.load(p)
    except (IOError, EOFError):
        return None


def dump_pickle(obj, name):
    os.makedirs(os.path.dirname(PICKLES_DIR), exist_ok=True)
    with open(PICKLES_DIR + name + ".pkl", "w+b") as p:
        pickle.dump(obj, p)


def load_metrics_for_config(base_model, preprocessing=False, augmentation=False, cv_type="cv10"):
    config_str = base_model + "_P" + str(preprocessing) + "_A" + str(augmentation) + "_"
    pkl = load_pickle(config_str + cv_type + "_metrics")
    return pkl
