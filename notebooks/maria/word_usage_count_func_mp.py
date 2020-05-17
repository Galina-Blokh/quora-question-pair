from collections import Counter
import pandas as pd
import time

def word_usage_count(df, ignore_register = False):
    """
    Receives a df with raw sentences and counts word usage
    Requires collections.Counter
    Args:
        df: dataframe with some questions
        ignore_register: boolean parameter to ignore register or not, defaults to False
    Returns:
        Counter object {word: number}.
    """
    questions1 = set(df.question1)
    questions2 = set(df.question2)
    unique_question_union = questions1.union(questions2)
    words_usage_count = Counter()
    for q in unique_question_union:
        for word in q.split():
            if ignore_register:
                words_usage_count[word.lower()]+= 1
            else:
                words_usage_count[word]+= 1
    return words_usage_count


def word_existence_in_corpus(df, ignore_register=False):
    """
    Receives a df with raw sentences and counts word usage
    Requires collections.Counter
    Args:
        df: dataframe with some questions
        ignore_register: boolean parameter to ignore register or not, defaults to False
    Returns:
        set of words occured in corpus from df.
    """
    return set(word_usage_count(df, ignore_register = ignore_register).elements())


df = pd.read_csv('train_dup.csv').drop(columns='id', axis=1)
df = df.dropna()
t0 = time.time()
total_word_dict = word_usage_count(df)
t1 = time.time()
print(len(total_word_dict))
print(t1 - t0)
print(len(word_existence_in_corpus(df)))