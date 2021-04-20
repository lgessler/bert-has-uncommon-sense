from pprint import pprint
from random import shuffle

import conllu
import sqlite3
from collections import Counter


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d
conn = sqlite3.connect('data/pdep/SQL/prepcorp.sqlite')
conn.row_factory = dict_factory

WHITELIST = [
    'about',
    'above',
    'across',
    'after',
    'against',
    'among',
    'around',
    'as',
    'at',
    'before',
    'behind',
    'below',
    'beneath',
    'beside',
    'besides',
    'between',
    'beyond',
    'circa',
    'despite',
    'during',
    'except',
    'for',
    'from',
    'including',
    'inside',
    'into',
    'near',
    'of',
    'off',
    'on',
    'onto',
    'over',
    'per',
    'since',
    'than',
    'through',
    'to',
    'toward',
    'towards',
    'under',
    'until',
    'unto',
    'up',
    'upon',
    'via',
    'with',
    'without'
]


rows = list(conn.execute('SELECT * FROM prepcorp'))
pprint(rows[0])

empty_token_dict = {field_name: "_" for field_name in conllu.parser.DEFAULT_FIELDS}
instances = []
for num_rows_visited, row in enumerate(rows):
    sense = row['sense']
    # Skip odd-looking senses
    if sense in ['unk', 'x', 'pv', 'adverb', '1(!)', '']:
        continue
    # Skip multiword prepositions like "out of"
    if ' ' in row['prep']:
        continue
    # This one's messed up
    if row['inst'] in [577203,]:
        continue
    # Only keep preps that are relatively common
    if row['prep'].lower() not in WHITELIST:
        continue

    # Index is char-based, so tokenization procedure needs to proceed by first excising the preposition
    prep_offset = row['preploc']
    prep_end = row['sentence'].find(' ', prep_offset+1)
    prep_token = row['sentence'][prep_offset:prep_end].strip()
    # Skip if we don't have a match (some offsets are unreliable)
    if prep_token.lower() != row['prep'].lower():
        continue
    # Now we can whitespace split everything to the left and right
    tokens_before_prep = row['sentence'][:prep_offset].strip().split()
    prep_index = len(tokens_before_prep)
    tokens = tokens_before_prep + [prep_token] + row['sentence'][prep_end + 1:].split()

    token_dicts = [empty_token_dict.copy() for t in tokens]
    for i, token in enumerate(tokens):
        token_dicts[i]['id'] = i + 1
        token_dicts[i]['form'] = token
    token_dicts[prep_index]['misc'] = {'Sense': sense}
    token_dicts[prep_index]['lemma'] = row['prep']

    # metadata
    source = row['source']
    instance_id = row['inst']

    tl = conllu.TokenList(token_dicts, {"source": str(source), "id": str(instance_id), "prep_id": str(prep_index + 1)})
    instances.append(tl.serialize())

shuffle(instances)
cutoff = int(len(instances) * 0.8)
with open('data/pdep/pdep_train.conllu', 'w') as f:
    f.write("".join(instances[:cutoff]))
with open('data/pdep/pdep_test.conllu', 'w') as f:
    f.write("".join(instances[cutoff:]))
