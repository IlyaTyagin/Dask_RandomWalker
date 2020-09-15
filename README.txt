DaskNode2Vec
Ilya Tyagin, Joey Liu

test

Required python dependencies:
  numba
  scipy
  dask
  distributed
  gensim
  numpy

Input:
  .mtx graph matrix
Output:
  graph embedding vectors in .emb format (gensim save_word2vec_format) and set of biased random walks

Graphs:
  Graphs karate.mtx and Reuters911.mtx are included in the archive. 
To get the 3rd test graph Bump_2911.mtx you can download it using this link: https://sparse.tamu.edu/Janna/Bump_2911

How to run the code:
  1) make sure that input .mtx graph matrix and the main.py script are in the same folder

  2) sample run:
    2.1) run Dask server using command:
      $ dask-ssh   --hostfile $PBS_NODEFILE
    2.2) run the script:
      pytnon main.py --dask_scheduler_node node1669:8786 --filename karate.mtx --walks_per_node_per_worker 3 --n_workers 4 --walklen 10 --p 1 --q 2 --compute_embeddings True --window_size 5 --dim 5 --iter 3

  3) parameters:
    dask_scheduler_node (str) - Dask server scheduler node
    filename (str) - .mtx matrix filename
    walks_per_node_per_worker (int) - number of walks per each starting node per each Dask worker
    n_workers (int) - number of Dask workers. If n_workers > available Dask workers, all Dask workers will be used
    walklen (int) - length of each random walk
    p (float) - parameter p (see the original paper)
    q (float) - parameter q (see the original paper)
    compute_embeddings (boolean) - perform Gensim Word2Vec learning phase. If it is not needed, just skip this and the following parameters:
    window_size (int) - window size (word2vec)
    dim (int) - embeddings dimensionality (word2vec)
    iter (int) - # of learning iterations (word2vec)
