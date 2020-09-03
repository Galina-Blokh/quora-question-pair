import random
import numpy as np
import torch

try:
    import dill as pickle
except:
    import pickle
import requests


def setup_seed(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def to_pickle(in_object, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(in_object, f, pickle.HIGHEST_PROTOCOL)


def from_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def translate_en_no_limits(x: str) -> str:
    try:
        url = "https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl=en&q=" + x
        headers = dict()
        headers["Accept"] = "*/*"
        headers["Host"] = "clients5.google.com"
        headers["Connection"] = "keep-alive"
        headers["Upgrade-Insecure-Requests"] = "1"
        headers[
            "User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36"
        headers["Content-Encoding"] = "gzip"
        headers["Accept-Language"] = "en-US,en;q=0.9,ru;q=0.8,he;q=0.7"
        headers["Content-Type"] = "application/json; charset=UTF-8"
        r = requests.get(url, verify=True, headers=headers)
        r.encoding = r.apparent_encoding
        json = r.json()
        return json['sentences'][0]['trans']
    except:
        return x
