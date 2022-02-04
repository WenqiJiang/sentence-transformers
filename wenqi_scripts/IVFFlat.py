import faiss
import time
import sys
import re
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

embedding_dim = 384
vec = np.fromfile('/data/computed_embeddings/realnewslike/c4-train.00306-of-00512', dtype='float32')
vec = vec.reshape(-1, embedding_dim)

print("Loaded vector shape: ", vec.shape)

# 500K as db vectors
# 1K as query vectors
# 10K as training vectors (heuristic: train vec = max(train_vec, 100 * centroid_num))

nb = 500 * int(1e3)
nq = 1 * int(1e3)
nt = 50 * int(1e3)

xb = vec[:nb]
xq = vec[nb: nb + nq]
xt = vec[nb + nq: nb + nq + nt]

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

print("Building Flat index for ground truth computation...")
index_flat = faiss.IndexFlat(embedding_dim)

i0 = 0
t0 = time.time()
for xs in matrix_slice_iterator(xb, 100000):
    i1 = i0 + xs.shape[0]
    print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
    sys.stdout.flush()
    index_flat.add(xs)
    i0 = i1

print("Searching for ground truth...")
D_gt, I_gt = index_flat.search(xq, 100)

n_dup = (D_gt[:, 0:1] == D_gt[:, 1:2]).sum()
print("\nOut of {} queries, {} have duplicated nearest neighbor (tied distance)".format(
    nq, n_dup))

def choose_train_size(index_key):

    # some training vectors for PQ and the PCA
    n_train = 256 * 1000

    if "IVF" in index_key:
        matches = re.findall('IVF([0-9]+)', index_key)
        ncentroids = int(matches[0])
        n_train = max(n_train, 100 * ncentroids)
    elif "IMI" in index_key:
        matches = re.findall('IMI2x([0-9]+)', index_key)
        nbit = int(matches[0])
        n_train = max(n_train, 256 * (1 << nbit))
    return n_train



print("Building IVFFlat for ANNS...")
nlist = 1024
index_key = "IVF{},Flat".format(nlist)
index_ivfflat = faiss.index_factory(embedding_dim, index_key)
#quantizer = faiss.IndexFlat(embedding_dim)
#index_ivfflat = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)

xtsub = xt[:choose_train_size(index_key)]
index_ivfflat.train(xtsub)
i0 = 0
t0 = time.time()
for xs in matrix_slice_iterator(xb, 100000):
    i1 = i0 + xs.shape[0]
    print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
    sys.stdout.flush()
    index_ivfflat.add(xs)
    i0 = i1

for i in range(10):
    nprobe = int(2 ** i)
    print("\nnprobe = ", nprobe)
    index_ivfflat.nprobe = nprobe
    topK = 100
    D, I = index_ivfflat.search(xq, topK)

    for rank in 1, 10, 100:
        if rank <= topK:
            n_ok = (I[:, :rank] == I_gt[:, :1]).sum()
            print("R@{} = {:.4f}".format(rank, (n_ok / float(nq))))
# print("Distance (ground truth): ", D_gt[:10])
#print("Indices (ground truth): ", I_gt[:10])



print("Building IVF-PQ (16 byte quantization) for ANNS...")
nlist = 1024
index_key = "IVF{},PQ16".format(nlist)
index_ivfpq = faiss.index_factory(embedding_dim, index_key)

xtsub = xt[:choose_train_size(index_key)]
index_ivfpq.train(xtsub)
i0 = 0
t0 = time.time()
for xs in matrix_slice_iterator(xb, 100000):
    i1 = i0 + xs.shape[0]
    print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
    sys.stdout.flush()
    index_ivfpq.add(xs)
    i0 = i1

for i in range(10):
    nprobe = int(2 ** i)
    print("\nnprobe = ", nprobe)
    index_ivfpq.nprobe = nprobe
    topK = 100
    D, I = index_ivfpq.search(xq, topK)

    for rank in 1, 10, 100:
        if rank <= topK:
            n_ok = (I[:, :rank] == I_gt[:, :1]).sum()
            print("R@{} = {:.4f}".format(rank, (n_ok / float(nq))))




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
