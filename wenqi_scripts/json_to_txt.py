import json
import os

from nltk.tokenize import sent_tokenize

lines = []

fname = 'c4-train.00000-of-01024'
dir_in = '/data/c4/en/'
dir_out = '/data/plain_c4/en/'

# Read json file (1 object per line)
with open(os.path.join(dir_in, fname + '.json')) as f:
    for json_line in f:
        paragraph = json.loads(json_line)['text']
        sentences = sent_tokenize(paragraph)
        sentences = [sent + '\n' for sent in sentences]
        lines += sentences

# remove the last empty line
if lines[-1] == '\n':
    lines.pop()

# Write lines as plain text
with open(os.path.join(dir_out, fname + '.txt'), 'w') as f:
    f.writelines(lines)

