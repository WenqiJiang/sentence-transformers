import faiss
import time
import sys
import re
import numpy as np

from multiprocessing.dummy import Pool as ThreadPool

embedding_dim = 384
vec = np.fromfile('../c4-train.00307-of-00512.npy', dtype='float32')
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
abs_max = np.amax([np.amax(xb), -np.amin(xb)])
min_99_perc, max_99_perc = np.percentile(xb, [1.0, 99.0])
abs_99_perc = np.amax([min_99_perc, max_99_perc])
print("Max absolute value in the database: ", abs_max)
print("99% absolute value in the database: ", abs_99_perc)

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

def add_to_index(index, vecs):
    i0 = 0
    t0 = time.time()
    for xs in matrix_slice_iterator(vecs, 100000):
        i1 = i0 + xs.shape[0]
        print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
        sys.stdout.flush()
        index.add(xs)
        i0 = i1

print("Building Flat index for ground truth computation...")
index_flat = faiss.IndexFlat(embedding_dim)
add_to_index(index_flat, xb)

print("Searching for ground truth...")
D_gt, I_gt = index_flat.search(xq, 100)

n_dup = (D_gt[:, 0:1] == D_gt[:, 1:2]).sum()
print("\nOut of {} queries, {} have duplicated nearest neighbor (tied distance)".format(
    nq, n_dup))

def verify_gt(I_gt, I, topK):

    for rank in 1, 10, 100:
        if rank <= topK:
            n_ok = (I[:, :rank] == I_gt[:, :1]).sum()
            print("R@{} = {:.4f}".format(rank, (n_ok / float(nq))))
    if topK > 100:
        n_ok = (I[:, :topK] == I_gt[:, :1]).sum()
        print("R@{} = {:.4f}".format(topK, (n_ok / float(nq))))
#####     fp 16     #####
# Only the database is compressed, the query is not compressed
xb_fp16_compressed = xb.astype('float16')

# When using the data, convert it back to fp32
xb_fp16 = xb_fp16_compressed.astype('float32')

for i in range(10):
    print("i = {i}\tfp32 square sum: {l32}\tfp16 square sum: {l16}".format(
        i=i, l32=np.sum(xb[i]**2), l16=np.sum(xb_fp16[i]**2)))

print("Building index using floating point 16...")
index_flat_fp16 = faiss.IndexFlat(embedding_dim)
add_to_index(index_flat_fp16, xb_fp16)

print("Searching...")
D_fp16, I_fp16 = index_flat_fp16.search(xq, 100)

verify_gt(I_gt, I_fp16, 100)


#####     int N-bits     #####
# Only the database is compressed, the query is not compressed

def quantization_recall_trial(nbits, scale_factor):
    print("Experiment int{nbits}... scale factor = {s}".format(nbits=nbits, s=scale_factor))
    N_bins = 2 ** nbits
    bins = np.arange(-N_bins/2, N_bins/2) / N_bins * scale_factor
    xb_intnbits_compressed = np.digitize(xb, bins, right=False)
    print("Before quantization: \n", xb[0])
    print("After quantization: \n", xb_intnbits_compressed[0])

    # When using the data, convert it back to fp32
    xb_intnbits = (xb_intnbits_compressed.astype('float32') - N_bins/2) / N_bins * scale_factor
    
    for i in range(10):
        print("i = {i}\tfp32 square sum: {l32}\tint{nbits} square sum: {l16}".format(
            i=i, nbits=nbits, l32=np.sum(xb[i]**2), l16=np.sum(xb_intnbits[i]**2)))
    
    print("Building...")
    index_flat_intnbits = faiss.IndexFlat(embedding_dim)
    # When using the data, convert it back to fp32
    add_to_index(index_flat_intnbits, xb_intnbits.astype('float32'))
    
    print("Searching...")
    D_intnbits, I_intnbits = index_flat_intnbits.search(xq, 100)
    
    verify_gt(I_gt, I_intnbits, 100)

quantization_recall_trial(nbits=8, scale_factor=abs_max)
quantization_recall_trial(nbits=8, scale_factor=abs_99_perc)
quantization_recall_trial(nbits=12, scale_factor=abs_max)
quantization_recall_trial(nbits=12, scale_factor=abs_99_perc)
quantization_recall_trial(nbits=16, scale_factor=abs_max)
quantization_recall_trial(nbits=16, scale_factor=abs_99_perc)
quantization_recall_trial(nbits=20, scale_factor=abs_max)
quantization_recall_trial(nbits=20, scale_factor=abs_99_perc)
#quantization_recall_trial(nbits=24)
#quantization_recall_trial(nbits=32)
