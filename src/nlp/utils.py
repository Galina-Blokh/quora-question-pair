import json
import requests

from src.nlp.stop_words import STOP_WORDS

try:
    import regex as re
except:
    import re


class RegexPattern:
    non_ascii_pattern = re.compile('([^\x00-\x7A])+')

    @staticmethod
    def contains(pattern, string):
        return pattern.search(string) is not None


def google_translate(sentence: str, lang: str = "en") -> str:
    try:
        url = f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl={lang}&q=" + sentence
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
        return sentence


def translate_sentence(tokenizer, sentence):
    return " ".join(t.text if t.text in STOP_WORDS or not RegexPattern.contains(RegexPattern.non_ascii_pattern,
                                                                                t.text) or t.is_digit or t.is_punct else "translation of " + translate(
        t.text) for t in
                    tokenizer(sentence))
