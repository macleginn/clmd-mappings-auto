import sqlite3
import json
from collections import defaultdict, Counter
from glob import glob

import pandas as pd
from simalign import SentenceAligner
from tqdm import tqdm

from UDLib import *


def get_words_and_idx_dicts(tree):
    idx2id = {}
    id2idx = {}
    words = []
    for word_id in tree.keys:
        if '-' in word_id:
            continue
        idx = len(words)
        words.append(tree.nodes[word_id].FORM)
        idx2id[idx] = word_id
        id2idx[word_id] = idx
    return words, idx2id, id2idx


def get_path_to_ancestor(node, ancestor, tree):
    result = []
    while node != ancestor:
        result.append(tree.nodes[node].DEPREL)
        node = tree.nodes[node].HEAD
    return result


def is_descendant(node1, node2, tree):
    while tree.nodes[node1].DEPREL != 'root':
        node1 = tree.nodes[node1].HEAD
        if node1 == node2:
            return True
    return False


def get_path(node1, node2, tree):
    if node1 == node2:
        return ['same_node']
    if is_descendant(node1, node2, tree):
        return get_path_to_ancestor(node1, node2, tree)
    elif is_descendant(node2, node1, tree):
        return list(reversed(get_path_to_ancestor(node2, node1, tree)))
    else:
        # Find an ancestor of node1 that is also
        # an ancestor of node2.
        ancestor = node1
        while not is_descendant(node2, ancestor, tree):
            ancestor = tree.nodes[ancestor].HEAD
        return get_path_to_ancestor(node1, ancestor, tree) + \
            list(reversed(get_path_to_ancestor(node2, ancestor, tree)))


# Preprocess the English PUD
en_trees = conllu2trees('PUD/en_pud-ud-test.conllu')
en_data = list(map(get_words_and_idx_dicts, en_trees))

myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

paths = [path for path in glob('PUD/*.conllu') if not path.startswith('PUD/en_')]

for path in paths:
    stats = defaultdict(Counter)
    print(path)
    lang_code = path[4:6]
    trg_trees = conllu2trees(path)
    trg_data = list(map(get_words_and_idx_dicts, trg_trees))
#    counter = 1
    for en_tree, en_data_triple, trg_tree, trg_data_triple in tqdm(list(zip(
            en_trees, en_data, trg_trees, trg_data))):
        en_words, en_idx2id, _ = en_data_triple
        trg_words, trg_idx2id, _ = trg_data_triple
        alignment = myaligner.get_word_aligns(en_words, trg_words)['inter']
        alignment_dict = { en_idx2id[head]: trg_idx2id[tail]
                           for head, tail
                           in alignment }
        # Only look at nodes whose parents were aligned
        # to something. Ignore function words.
        for en_node, trg_node in alignment_dict.items():
            if en_tree.nodes[en_node].HEAD in alignment_dict:
                en_node_parent = en_tree.nodes[en_node].HEAD
            else:
                continue
            if en_tree.nodes[en_node].DEPREL not in [
                    'nsubj', 'obj', 'obl', 'iobj',
                    'advmod', 'ccomp', 'xcomp', 'advcl', 'acl', 'ccubj',
                    'amod', 'nmod', 'appos', 'nummod', 'compound'
            ]:
                continue
            deprel = en_tree.nodes[en_node].DEPREL
            trg_head = alignment_dict[en_node_parent]
            trg_path = get_path(trg_node, trg_head, trg_tree)
            stats[deprel]['+'.join(trg_path)] += 1
#        if counter == 100:
#            break
#        counter += 1
    # Prune rare mappings
    for deprel in stats:
        tmp_counts = { trg_path: count for trg_path, count
                       in stats[deprel].items()
                       if count >= 5 }
        stats[deprel] = tmp_counts
    pd.DataFrame(stats).fillna(0.0).T.to_csv(f'en-{lang_code}_mapping_stats.csv')
