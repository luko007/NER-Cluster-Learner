import os
import numpy as np
import pandas as pd
from nltk import pos_tag

TAG_COLUMN = 'Tag'
WORD_COLUMN = 'Word'
POS_COLUMN = 'POS'


def load_broad_twitter_corpus():
    """
    Loads the file broad_twitter_corpus.json as DataFrame and adds a column of POS tags using NLTK library
    """
    broad_twitter_corpus_path = os.path.join('Data', 'broad_twitter_corpus')
    broad_twitter_corpus_files = ['a', 'b', 'e', 'f', 'g', 'h']
    all_entities, all_tokens, all_pos = [], [], []

    for f_name in broad_twitter_corpus_files:
        df = pd.read_json(os.path.join(broad_twitter_corpus_path, f_name + '.json'), lines=True)
        new_entities, new_tokens = json_to_ner_df(df)

        all_entities.extend(new_entities)
        all_tokens.extend(new_tokens)

    for empty_item in sorted(np.where(np.array(all_tokens) == '')[0], reverse=True):
        all_tokens.pop(empty_item)
        all_entities.pop(empty_item)

    all_pos = tag_text_column(all_tokens)
    df = pd.DataFrame({WORD_COLUMN: all_tokens,
                             TAG_COLUMN: all_entities,
                             POS_COLUMN: all_pos})
    return df


def json_to_ner_df(df: pd.DataFrame) -> (list, list):
    """
    Returns a new DataFrame from a json typed DataFrame.
    """
    df = df[['entities', 'tokens']].to_dict(orient='list')
    new_entities = []
    new_tokens = []
    for entity, token in zip(df['entities'], df['tokens']):
        for e, t in zip(entity, token):
            new_entities.append(e)
            new_tokens.append(t)
    return new_entities, new_tokens


def tag_text_column(tokens: list) -> list:
    """
    Given a list of tokens, returns their respected POS tags.
    """
    all_pos = pos_tag(tokens)
    all_pos = [p[1] for p in all_pos]
    return all_pos
