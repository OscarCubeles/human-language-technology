import pandas as pd

def load_files(path_scores, path_sents):
    scores = []
    for l in open(path_scores, 'r'):
        scores.append(float(l.strip()))
    sent_pairs = []
    for l in open(path_sents, 'r', encoding='utf-8'):
        sent_pairs.append(l.strip().split('\t'))
    assert len(scores) == len(sent_pairs), "Scores and sentence pairs must have the same length."
    df = pd.DataFrame(sent_pairs, columns=['sentence1', 'sentence2'])
    df['score'] = scores
    return df

def merge_datasets(dataframes):
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df
