#!/usr/bin/env python3

import numba
from numba import jit
import numpy as np
import os
from scipy import sparse
import scipy.io
import time

import dask.bag as dbag
import distributed 

import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--dask_scheduler_node', type = str)
parser.add_argument('--filename', type = str)
parser.add_argument('--walks_per_node_per_worker', type = int)
parser.add_argument('--n_workers', type = int, default = 10)
parser.add_argument('--walklen', type = int, default = 10)
parser.add_argument('--p', type = float, default = 2)
parser.add_argument('--q', type = float, default = 2)
parser.add_argument('--compute_embeddings', type = bool, default = False)
parser.add_argument('--window_size', type = int, default = 5)
parser.add_argument('--dim', type = int, default = 64)
parser.add_argument('--iter', type = int, default = 10)
args = parser.parse_args()

p = args.p
q = args.q

print('args:')
print(f'scheduler node: {args.dask_scheduler_node}')
print(f'matrix filename: {args.filename}')
print('---Walker parameters---')
print(f'walks per node per worker: {args.walks_per_node_per_worker}')
print(f'walklen: {args.walklen}')
print(f'p: {p}')
print(f'q: {q}')
print('---Word2Vec parameters:---')
print(f'compute embeddings: {args.compute_embeddings}')
if args.compute_embeddings:
  print(f'window size: {args.window_size}')
  print(f'embeddings dimensionality: {args.dim}')
  print(f'number of iterations: {args.iter}')

daskClient = distributed.Client(args.dask_scheduler_node)
filename = args.filename

n_workers = args.n_workers
n_workers_real = len(daskClient.scheduler_info()['workers'])
if n_workers > n_workers_real:
  n_workers = n_workers_real
print(f'\nStarting...\n# of Dask workers: {n_workers}')

@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _csr_random_walk(Tdata, Tindptr, Tindices,
                    sampling_nodes,
                    walklen, p_prob, q_prob):
    """
    Create random walks from the transition matrix of a graph 
        in CSR sparse format

    NOTE: scales linearly with threads but hyperthreads don't seem to 
            accelerate this linearly
    
    Parameters
    ----------
    Tdata : 1d np.array
        CSR data vector from a sparse matrix. Can be accessed by M.data
    Tindptr : 1d np.array
        CSR index pointer vector from a sparse matrix. 
        Can be accessed by M.indptr
    Tindices : 1d np.array
        CSR column vector from a sparse matrix. 
        Can be accessed by M.indices
    sampling_nodes : 1d np.array of int
        List of node IDs to start random walks from.
        Is generally equal to np.arrange(n_nodes) repeated for each epoch
    walklen : int
        length of the random walks

    Returns
    -------
    out : 2d np.array (n_walks, walklen)
        A matrix where each row is a random walk, 
        and each entry is the ID of the node
    """
    n_walks = len(sampling_nodes)
    res = np.empty((n_walks, walklen), dtype=np.int64)
    for i in numba.prange(n_walks):
      # if not i%100000:
        #  print(f'{i} of {n_walks} done...')
        # Current node (each element is one walk's state)
        state = sampling_nodes[i]
        for k in range(walklen-1):
            if k == 0:
                # Write state
                res[i, k] = state
                # Find row in csr indptr
                start = Tindptr[state]
                end = Tindptr[state+1]
                # transition probabilities
                p = Tdata[start:end]
                # cumulative distribution of transition probabilities
                cdf = np.cumsum(p)
                # Random draw in [0, 1] for each row
                # Choice is where random draw falls in cumulative distribution
                draw = np.random.rand()
                # Find where draw is in cdf
                # Then use its index to update state
                next_idx = np.searchsorted(cdf, draw)
                # Winner points to the column index of the next node
                state = Tindices[start + next_idx]
            else:
                # Write state
                res[i, k] = state
                # previous state
                previous_state = res[i, k-1]
                # Find row in csr indptr
                start = Tindptr[state]
                end = Tindptr[state+1]
                previous_start = Tindptr[previous_state]
                previous_end = Tindptr[previous_state+1]
                
                # find neighbors
                a = Tindices[start:end]
                b = Tindices[previous_start:previous_end]
                p = np.copy(Tdata[start:end])
                
                # biased random walk
                c = 0
                d = 0
                while c < len(a) and d <= len(b):
                    if a[c] == previous_state:
                        p[c] = 1/p_prob # return parameter
                        c += 1
                    else:
                        if a[c] < b[d]:
                            p[c] = 1/q_prob # in-out parameter
                            c += 1
                        elif a[c] == b[d]:
                            p[c] = 1
                            c += 1
                            d += 1
                        else:
                            d += 1
                    if d == len(b):
                        while c < len(a):
                            p[c] = 1/q_prob # in-out parameter
                            c += 1
                
                # normalize and get next state
                cdf = np.cumsum(np.divide(p, np.sum(p)))
                draw = np.random.rand()
                next_idx = np.searchsorted(cdf, draw)
                state = Tindices[start + next_idx]
                
        # Write final states
        res[i, -1] = state
        
    return res

def wrapper_csr_random_walk(T, walklen = 5, walksPerNode = 3, p = 2, q = 2):
    Tdata = T.data
    Tindptr = T.indptr
    Tindices = T.indices
    sampling_nodes = np.repeat(np.arange(T.shape[0]), walksPerNode)
    print('performing walks...')
    startTime = time.time()
    walks = _csr_random_walk(Tdata, Tindptr, Tindices,
        sampling_nodes, walklen, p , q)
    print(f'Worker finished. Walking time: {time.time() - startTime}s')
    return walks

def walksToString(walks):
  return '\n'.join([' '.join(str(n) for n in p) for p in walks]) 

def openMatrix(filename):
  startTime = time.time()
  matrix = scipy.io.mmread(filename).tocsr().astype('float64')
  print(f'opened matrix size: {matrix.shape}\nOpening time: {time.time() - startTime}')
  return matrix

def check_symmetric(a, rtol=1e-05, atol=1e-08):
      return np.allclose(a, a.T, rtol=rtol, atol=atol)

tempFolderPath = filename.split('.')[0] + '_temp'
flag = 0
if not os.path.exists(tempFolderPath):
  os.mkdir(tempFolderPath)
  flag = 1
else:
  print(f'\n CURRENT FOLDER: {os.getcwd()}\nSorry, there is another temp folder for this graph.\nDo you want to remove the old temp folder for this graph? [y/n]')
  userInput = input()
  while userInput not in 'yn' :
    print('Please enter the correct symbol')
    userInput = input()
  if userInput == 'y':
    shutil.rmtree(tempFolderPath)
    flag = 1
  else:
    flag = 0


if flag:
  print('Performing walks...')
  start_time = time.time()

  mtr_dbag = dbag \
  .from_sequence(
      [os.getcwd() + '/' + filename]*n_workers
      ) \
  .map(openMatrix) \
  .map( 
      wrapper_csr_random_walk, 
      walklen = args.walklen, 
      walksPerNode = args.walks_per_node_per_worker, 
      p = p, 
      q = q
      ) \
  .map(walksToString) \
  .to_textfiles(
      tempFolderPath+'/'+filename+'_walk_*.txt'
      )
  print(f'Finished.\nTime used: {time.time() - start_time}s.')

def ReadWalksFromFiles(folder):
  corpus = []
  for filename in os.listdir(folder):
    file = open(folder + '/' + filename, encoding="utf8").read()
    file = [[n for n in walk.split()] for walk in file.split('\n')]
    corpus += file
  return corpus

from gensim.models import Word2Vec
def learn_embeddings(walks, output):
   '''
   Learn embeddings by optimizing the Skipgram objective using SGD.
   '''
   model = Word2Vec(
     walks,
     size=args.dim,
     window=args.window_size,
     min_count=0,
     sg=1,
     iter=args.iter)
   model.wv.save_word2vec_format(output)

if args.compute_embeddings:
  print('Computing embeddings using Gensim...')
  gensTime = time.time()
  w2v = dbag.from_sequence([os.getcwd() + '/' + tempFolderPath]) \
  .map(ReadWalksFromFiles) \
  .map(learn_embeddings, output = os.getcwd() + '/' + filename + '.emb') \
  .compute()
  print(f'Gensim learning time: {time.time() - gensTime}s')

print('finished')
