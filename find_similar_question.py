import numpy
import pickle
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import train_models

def prepare_search_engine(df):


    # Dimension of our vector space
    dimension = len(df['ft_vector_q1'][0])
    print('dimension: ', dimension)

    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections('rbp', 14)

    # Create engine with pipeline configuration
    engine = Engine(dimension, lshashes=[rbp])

    # Index 1000000 random vectors (set their data to a unique string)
    for index in range(len(df)):
        v = df['ft_vector_lem_q1'][index]

        engine.store_vector(v, index)
    return engine

if __name__ == '__main__':
    df = pickle.load(open('single_feats.pkl', 'rb'))
    print(df.head())
    print(df.columns)
    engine = prepare_search_engine(df)
    sfe = train_models.SingleFeatsExctractor('models/small_en.bin')
    question = 'Are movies on YouTube censored?'
    question_feats = sfe.exctract_feats_from_text(question)
    # Get nearest neighbours
    N = engine.neighbours(question_feats['ft_vector_lem'])
    # print(N.shape)
    for neighbour in N:
        print(df['question1_q1'][neighbour[1]])

"""
create df:
- df['question1_q1'][neighbour[1]]) - all the neighbour questions + all single features
+ question (as q2) + single features
"""