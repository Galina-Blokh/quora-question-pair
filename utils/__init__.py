import pickle


def to_pickle(in_object, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(in_object, f, pickle.HIGHEST_PROTOCOL)


def from_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)