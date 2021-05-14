import os
import pickle


def pickle_read(path):
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def pickle_write(o, path):
    dirs = os.sep.join(path.split(os.sep)[:-1])
    if len(dirs) > 0:
        os.makedirs(dirs, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(o, f)
