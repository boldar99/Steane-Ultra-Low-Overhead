import numpy as np
from collections import deque

# target H (3 x 7)
H = np.array([
    [1,1,1,1,0,0,0],
    [0,1,1,0,1,1,0],
    [0,0,1,1,0,1,1]
], dtype=np.int8)

def mat_to_int(M):
    # flatten row-major into 21-bit integer
    arr = M.reshape(-1)
    val = np.int32(0)
    int32_1 = np.int32(1)
    for k,b in enumerate(arr):
        if b:
            val |= (int32_1 << k)
    return val

def int_to_mat(x):
    arr = [(x>>k)&1 for k in range(21)]
    return np.array(arr, dtype=np.uint8).reshape((3,7))

# precompute toggle masks for single-entry flips
toggle_masks = [1<<k for k in range(21)]

def row_add(x, i, j):
    # add row i to row j (row_j ^= row_i) on integer representation
    row_i = (x >> (i*7)) & ((1<<7)-1)
    mask = row_i << (j*7)
    return x ^ mask

zero = 0
H_int = mat_to_int(H)

def bfs_min_distance(depth_limit=11):
    if zero == H_int:
        return 0, []
    visited = {zero}
    parent = {zero: None}
    parent_op = {zero: None}
    q = deque([zero])
    depth = 0
    while q and depth < depth_limit:
        for _ in range(len(q)):
            cur = q.popleft()
            # toggles
            for k in range(21):
                nb = cur ^ toggle_masks[k]
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    parent_op[nb] = ('toggle', k)
                    if nb == H_int:
                        # reconstruct path
                        path = []
                        node = nb
                        while parent[node] is not None:
                            path.append(parent_op[node])
                            node = parent[node]
                        path.reverse()
                        return depth+1, path
                    q.append(nb)
            # row-adds
            for i in range(3):
                for j in range(3):
                    if i==j: continue
                    nb = row_add(cur, i, j)
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = cur
                        parent_op[nb] = ('rowadd', i, j)
                        if nb == H_int:
                            path = []
                            node = nb
                            while parent[node] is not None:
                                path.append(parent_op[node])
                                node = parent[node]
                            path.reverse()
                            return depth+1, path
                        q.append(nb)
        depth += 1
    return None, None

if __name__ == "__main__":
    # check up to depth 10 (should be None), then find depth 11 solution
    d10, _ = bfs_min_distance(depth_limit=10)
    print("Distance <= 10 found?:", d10 is not None)
    d11, path = bfs_min_distance(depth_limit=11)
    print("Shortest distance found (<=11):", d11)
    if path:
        print("Path (operations):")
        for op in path:
            print(op)
