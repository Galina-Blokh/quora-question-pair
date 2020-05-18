from collections import Counter
import pandas as pd
import time


def word_usage_count(df, column1, column2, ignore_register = False):
    """
    Receives a df with raw sentences and counts word usage
    Requires collections.Counter
    Args:
        df: dataframe with some questions
        column1: columns to unite and count word usage
        column2: columns to unite and count word usage
        ignore_register: boolean parameter to ignore register or not, defaults to False
    Returns:
        Counter object {word: number}.
    """
    questions1 = set(df[column1])
    questions2 = set(df[column2])
    unique_question_union = questions1.union(questions2)
    words_usage_count = Counter()
    for q in unique_question_union:
        for word in q.split():
            if ignore_register:
                words_usage_count[word.lower()]+= 1
            else:
                words_usage_count[word]+= 1
    return words_usage_count

def word_existence_in_corpus(df,column1, column2, ignore_register=False):
    """
    Receives a df with raw sentences and counts word usage
    Requires collections.Counter
    Args:
        df: dataframe with some questions
        column1: columns to unite and get words
        column2: columns to unite and get words
        ignore_register: boolean parameter to ignore register or not, defaults to False
    Returns:
        set of words occured in corpus from df.
    """
    return set(word_usage_count(df, column1, column2, ignore_register = ignore_register).elements())


df = pd.read_csv('train_dup.csv').drop(columns='id', axis=1)
df = df.dropna()
t0 = time.time()
total_word_dict = word_usage_count(df,'question1', 'question2',)
t1 = time.time()
print(len(total_word_dict))
print(t1 - t0)
print(len(word_existence_in_corpus(df,'question1', 'question2',)))
