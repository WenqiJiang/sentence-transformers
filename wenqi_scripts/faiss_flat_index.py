import faiss
import time
import sys
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

embedding_dim = 384
vec = np.fromfile('../data/computed_embeddings/realnewslike/c4-train.00000-of-0051.data', dtype='float32')
vec = vec.reshape(-1, embedding_dim)

print("Loaded vector shape: ", vec.shape)

# 500K as db vectors
# 1K as query vectors
# 10K as training vectors (heuristic: train vec = max(train_vec, 100 * centroid_num))

xb = vec[:500 * int(1e3)]
xq = vec[500 * int(1e3): 501 * int(1e3)]
xt = vec[501 * int(1e3): 511 * int(1e3)]

print("Shape: DB {} \t Query {} \t Training {}".format(
    xb.shape, xq.shape, xt.shape))

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

index = faiss.IndexFlat(embedding_dim)

i0 = 0
t0 = time.time()
for xs in matrix_slice_iterator(xb, 100000):
    i1 = i0 + xs.shape[0]
    print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
    sys.stdout.flush()
    index.add(xs)
    i0 = i1

print("Searching...")
D_gt, I_gt = index.search(xq, 100)
print("Distance (ground truth): ", D_gt[:10])
print("Indices (ground truth): ", I_gt[:10])





"""
def get_trained_index():
    filename = "%s/%s_%s_trained.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = faiss.index_factory(d, index_key)

        n_train = choose_train_size(index_key)

        xtsub = xt[:n_train]
        print("Keeping %d train vectors" % xtsub.shape[0])
        # make sure the data is actually in RAM and in float
        xtsub = xtsub.astype('float32').copy()
        index.verbose = True

        t0 = time.time()
        index.train(xtsub)
        index.verbose = False
        print("train done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index
"""


"""
def get_populated_index():

    filename = "%s/%s_%s_populated.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = get_trained_index()
        i0 = 0
        t0 = time.time()
        for xs in matrix_slice_iterator(xb, 100000):
            i1 = i0 + xs.shape[0]
            print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
            sys.stdout.flush()
            index.add(xs)
            i0 = i1
        print()
        print("Add done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
        if save_numpy_index:
            print("Saving index to numpy array...")
            chunk = faiss.serialize_index(index)
            np.save("{}.npy".format(filename), chunk)
            print("Finish saving numpy index")
    return index
"""
