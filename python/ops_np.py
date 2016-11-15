import numpy as np

def make_batches(X, y, batch_size=1):
    indices = np.arange(X.shape[0])
    length = len(indices)
    np.random.shuffle(indices)
        
    for ndx in range(0, length, batch_size):
        curr_idxs = indices[ndx:min(ndx + batch_size, length)]
        yield (np.array(X[curr_idxs]), np.array(y[curr_idxs]))
