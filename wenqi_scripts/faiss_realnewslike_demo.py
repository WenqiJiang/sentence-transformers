"""
This script import the 64 partitions of the realnewslike sentences,
    feed them into faiss Flat index, and creat the mapping between 
    faiss index and the (file name, line number)
"""

import faiss
import time
import sys
import os
import numpy as np
import pickle

from multiprocessing.dummy import Pool as ThreadPool
from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('/home/wejiang/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2')

embedding_dim = 384
num_files = 60

# 500K as db vectors
# 1K as query vectors
# 10K as training vectors (heuristic: train vec = max(train_vec, 100 * centroid_num))

# xb = vec[:500 * int(1e3)]
# xq = vec[500 * int(1e3): 501 * int(1e3)]
# xt = vec[501 * int(1e3): 511 * int(1e3)]

# print("Shape: DB {} \t Query {} \t Training {}".format(
#     xb.shape, xq.shape, xt.shape))

def rate_limited_imap(f, l):
    'a thread pre-processes the next element'
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()

def matrix_slice_iterator(x, bs):
    " iterate over the lines of x in blocks of size bs"
    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    return rate_limited_imap(
        lambda i01: x[i01[0]:i01[1]].astype('float32').copy(),
        block_ranges)

def save_obj(obj, dirc, name):
    # note use "dir/" in dirc
    with open(os.path.join(dirc, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, protocol=4) # for py37,pickle.HIGHEST_PROTOCOL=4

def load_obj(dirc, name):
    with open(os.path.join(dirc, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def map_faiss_idx_to_file_and_line(idx, ID_dict):
    """
    return the File ID and line number given the faiss returned index
        the ID_dict format: key = file ID, value = (start line ID, number of lines in this file)
        the start line ID is the accumulated line number in previous files
    """
    keys = []
    for key in ID_dict:
        keys.append(key)
    keys = sorted(keys)
    for key in keys:
        start_line_ID = ID_dict[key][0]
        num_lines = ID_dict[key][1]
        if idx >= start_line_ID and idx < start_line_ID + num_lines:
            file_ID = key
            line_ID = idx - start_line_ID
            return (file_ID, line_ID)
    print("Did not find the mapping in the dictionary")
    raise ValueError

def get_line_from_file(fname, line_ID):
    """ return the line_ID th line from file (dir name) """
    with open(fname) as f:
        # read the content of the file opened
        content = f.readlines()
        # read line_ID th line from the file
        return content[line_ID]


index = faiss.IndexFlat(embedding_dim)

tmpdir = '../data/faiss_indexes'
if not os.path.exists(tmpdir):
    os.mkdir(tmpdir)
index_key = 'Flat'
dbname = 'realnewslike_{}_files_{}'.format(num_files, index_key)
index_parent_dir = os.path.join(tmpdir, dbname)
if not os.path.exists(index_parent_dir):
    os.mkdir(index_parent_dir)

index_dir = os.path.join(index_parent_dir, dbname + '_populated.index')
ID_dict_fname = dbname + '_ID_map'
ID_dict_dir = os.path.join(index_parent_dir, ID_dict_fname)

if not os.path.exists(index_dir):
    print("Index does not exist, creating index...")
    sys.stdout.flush()
    i0 = 0
    t0 = time.time()
    ID_dict = dict()
    for n in range(num_files):
        vec = np.fromfile('../data/computed_embeddings/realnewslike/c4-train.00{}-of-00512.data'.format(str(n).zfill(3)), dtype='float32')
        vec = vec.reshape(-1, embedding_dim)
        print("Loaded {} th file, vector shape: {}".format(n, vec.shape))
        if not ID_dict: # empty
            ID_dict[n] = (0, vec.shape[0])
        else:
            ID_dict[n] = (ID_dict[n - 1][0] + ID_dict[n - 1][1], vec.shape[0])
        for xs in matrix_slice_iterator(vec, 100000):
            i1 = i0 + xs.shape[0]
            print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
            sys.stdout.flush()
            index.add(xs)
            i0 = i1
    faiss.write_index(index, index_dir)
    save_obj(ID_dict, index_parent_dir, ID_dict_fname)
elif not os.path.exists(ID_dict_dir + '.pkl'):
    print("Mapping dictionary does not exist, creat mapping...")
    sys.stdout.flush()
    ID_dict = dict()
    for n in range(num_files):
        vec = np.fromfile('../data/computed_embeddings/realnewslike/c4-train.00{}-of-00512.data'.format(str(n).zfill(3)), dtype='float32')
        vec = vec.reshape(-1, embedding_dim)
        print("Loaded {} th file, vector shape: {}".format(n, vec.shape))
        if not ID_dict: # empty
            ID_dict[n] = (0, vec.shape[0])
        else:
            ID_dict[n] = (ID_dict[n - 1][0] + ID_dict[n - 1][1], vec.shape[0])
    save_obj(ID_dict, index_parent_dir, ID_dict_fname)
else:
    print("loading", index_dir)
    sys.stdout.flush()
    index = faiss.read_index(index_dir)
    ID_dict = load_obj(index_parent_dir, dbname + '_ID_map')

# The MS MACRO query part
print("\n\n==== Start querying using MS MACRO =====\n\n")
sys.stdout.flush()


sentences = [
    'does xpress bet charge to deposit money in your account',
    'how much is a cost to run disneyland',
    'botulinum definition',
    'do physicians pay for insurance from their salaries?',
    'here there be dragons comic',
    'blood diseases that are sexually transmitted',
    'define bona fides',
    'effects of detox juice cleanse',
    'do prince harry and william have last names',
    'can hives be a sign of pregnancy',
    'causes of petechial hemorrhage',
    'how long does it take to get your bsrn if you already have a bachelors degree',
    'symptoms of ptsd in vietnam veterans',
    'how coffee works quote',
    'does contraction of the ciliary muscles shorten the lens',
    'do spiders eat other animals',
    'how long is the flight from chicago to cairo',
    'how titanic facts',
    'how to play blu ray discs',
    'central city definition']

sentence_embeddings = model.encode(sentences)

print("Searching...")
D_gt, I_gt = index.search(sentence_embeddings, 100)
print("Distance (ground truth): ", D_gt[:10])
print("Indices (ground truth): ", I_gt[:10])

for q in range(10):
    print("The nearest neighbor of the {} th query: ".format(q))
    print("Query contents:\n", sentences[q])
    print("Nearest neighbors:")
    for i in range(10):
        file_ID, line_ID = map_faiss_idx_to_file_and_line(I_gt[q][i], ID_dict)
        print("i = {}\tFile ID = {}\tLine ID = {}".format(i, file_ID, line_ID))
        line = get_line_from_file('../data/plain_c4/realnewslike/c4-train.00{}-of-00512.txt'.format(str(file_ID).zfill(3)), line_ID)
        print(line)

# The CC part
print("\n\n==== Start querying using CC realnewslike corpus =====\n\n")
sys.stdout.flush()

vec = np.fromfile('../data/computed_embeddings/realnewslike/c4-train.00063-of-00512.data', dtype='float32')
vec = vec.reshape(-1, embedding_dim)
xq = vec[0:10]

print("Searching...")
D_gt, I_gt = index.search(xq, 100)
print("Distance (ground truth): ", D_gt[:10])
print("Indices (ground truth): ", I_gt[:10])

for q in range(10):
    print("The nearest neighbor of the {} th query: ".format(q))
    print("Query contents:")
    print(get_line_from_file('../data/plain_c4/realnewslike/c4-train.00063-of-00512.txt', q))
    print("Nearest neighbors:")
    for i in range(10):
        file_ID, line_ID = map_faiss_idx_to_file_and_line(I_gt[q][i], ID_dict)
        print("i = {}\tFile ID = {}\tLine ID = {}".format(i, file_ID, line_ID))
        line = get_line_from_file('../data/plain_c4/realnewslike/c4-train.00{}-of-00512.txt'.format(str(file_ID).zfill(3)), line_ID)
        print(line)

