from __future__ import annotations
import argparse, os, time, json, pickle, gzip
from collections import deque, Counter
from tqdm import tqdm

import numpy as np

# --- Problem definition (same encoding used earlier) ---
H_rows = (
    (1,1,1,1,0,0,0),
    (0,1,1,0,1,1,0),
    (0,0,1,1,0,1,1),
)

def row_to_mask(row):
    mask = 0
    for i,b in enumerate(row):
        if b:
            mask |= (1 << i)
    return mask


def rows_to_state(rows):
    return tuple(row_to_mask(row) for row in rows)
ZERO = (0,0,0)
H_state = rows_to_state(H_rows)
col_masks = [1<<i for i in range(7)]

# primitives: data->ancilla then ancilla->ancilla
ops = []
for d in range(7):
    for a in range(3):
        ops.append(("d2a", d, a))
for c in range(3):
    for t in range(3):
        if c != t:
            ops.append(("a2a", c, t))
NUM_OPS = len(ops)  # 27
SHORT_LEN = 11



STEANE_STABS = []
for i in range(8):
    I = 0
    if i >> 0 & 1:
        I ^= row_to_mask(H_rows[0])
    if i >> 1 & 1:
        I ^= row_to_mask(H_rows[1])
    if i >> 2 & 1:
        I ^= row_to_mask(H_rows[2])
    STEANE_STABS.append(I)


def apply_op(state, op):
    s0,s1,s2 = state
    if op[0] == "d2a":
        d,a = op[1], op[2]
        if a==0: s0 ^= col_masks[d]
        elif a==1: s1 ^= col_masks[d]
        else: s2 ^= col_masks[d]
    else:
        c,t = op[1], op[2]
        rows=[s0,s1,s2]
        rows[t] ^= rows[c]
        s0,s1,s2 = rows
    return (s0,s1,s2)


# human readable op string
def op_to_str(op):
    if op[0]=='d2a': return f"d2a(d{op[1]}->a{op[2]})"
    return f"a2a(a{op[1]}->a{op[2]})"


# --- BFS dist map (fast) ---
def compute_bfs():
    dist = {ZERO:0}
    q = deque([ZERO])
    found_dist = None
    while q:
        u = q.popleft()
        du = dist[u]
        for op in ops:
            v = apply_op(u, op)
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
                if v == H_state and found_dist is None:
                    found_dist = du+1
    return found_dist, dist


# --- partial generation (layer-limited enumeration) ---
def generate_partials(start_state, depth, out_dir, resume=False):
    os.makedirs(out_dir, exist_ok=True)
    # layer: mapping state -> list of byte-seqs
    layer = { start_state: [ bytes() ] }
    for d in range(1, depth+1):
        print(f"[partials] expanding depth {d} (states: {len(layer)})")
        next_layer = {}
        for state, blist in tqdm(layer.items(), desc=f"layer-{d}", unit="state"):
            for b in blist:
                for op_idx in range(NUM_OPS):
                    nxt = apply_op(state, ops[op_idx])
                    nb = b + bytes([op_idx])
                    next_layer.setdefault(nxt, []).append(nb)
        if d < depth:
            layer = next_layer
        else:
            # write per-middle-state files
            print(f"[partials] writing {len(next_layer)} middle-state files to {out_dir}")
            for state, blist in tqdm(next_layer.items(), desc="writing"):
                fname = f"st_{state[0]}_{state[1]}_{state[2]}.pkl"
                pth = os.path.join(out_dir, fname)
                if resume and os.path.exists(pth):
                    continue
                with open(pth, "wb") as f:
                    pickle.dump(blist, f, protocol=pickle.HIGHEST_PROTOCOL)
            return list(next_layer.keys())
    return []


# --- reconstruct generator (returns op-index lists) ---
def mitm_reconstruct_generator(forward_dir, backward_dir):
    f_files = set(os.listdir(forward_dir))
    b_files = set(os.listdir(backward_dir))
    common = sorted(list(f_files & b_files))
    for fname in common:
        fpath = os.path.join(forward_dir, fname)
        bpath = os.path.join(backward_dir, fname)
        with open(fpath, "rb") as f:
            f_list = pickle.load(f)
        with open(bpath, "rb") as f:
            b_list = pickle.load(f)
        for fb in f_list:
            f_seq = list(fb)
            for bb in b_list:
                seq = f_seq + list(bb)[::-1]
                if len(seq) == SHORT_LEN:
                    yield seq

def compute_weight2_locations_from_seq(seq_ops):
    L = len(seq_ops)
    out = []

    # Precompute, for each ancilla, the set of gap positions p (0..L-1) where op at index p touches ancilla
    ancilla_positions = {0: list(), 1: list(), 2: list()}
    for p in range(0, L):
        op = ops[ seq_ops[p] ]
        if op[0] == "d2a":
            _, d, a = op
            ancilla_positions[a].append(p)
        else:  # a2a
            _, c, t = op
            ancilla_positions[c].append(p)
            ancilla_positions[t].append(p)
    for i in range(3):
        ancilla_positions[i].sort()
        ancilla_positions[i] = ancilla_positions[i][1:]

    # Now for each ancilla, only test the gaps before ops that touch it
    for a0 in range(3):
        interesting_ps = ancilla_positions[a0]
        for p in interesting_ps:
            S = 1 << a0
            D = 0
            for k in range(p, L):
                op = ops[ seq_ops[k] ]
                if op[0] == "a2a":
                    c, t = op[1], op[2]
                    if (S >> t) & 1:
                        S ^= (1 << c)
                else:
                    d, a = op[1], op[2]
                    if (S >> a) & 1:
                        D ^= (1 << d)

            min_weight = 7
            min_weight_error = D
            for stab in STEANE_STABS:
                N = bin(D ^ stab).count("1")
                if N < min_weight:
                    min_weight = N
                    min_weight_error = D ^ stab
                    if min_weight < 2:
                        break
            if min_weight < 2:
                continue
            data_bits = tuple(i for i in range(7) if (min_weight_error >> i) & 1)
            out.append({"ancilla": a0, "position": p, "data_bits": data_bits})
    return out

# --- detector simulation (exact) ---
def detector_detects_fault(seq_ops, a0, p, T, q):
    if q <= p: return False
    S = 1 << a0
    for k in range(p, q):
        op = ops[ seq_ops[k] ]
        if op[0] == "a2a":
            c,t = op[1], op[2]
            if (S >> t) & 1: S |= (1 << c)
    return ((S >> T) & 1) != 0

def candidate_detects_all(seq_ops, weight2_list, candidate):
    for ent in weight2_list:
        a0 = ent["ancilla"]; p = ent["position"]
        detected = False
        for (T,q) in candidate:
            if detector_detects_fault(seq_ops, a0, p, T, q):
                detected = True
                break
        if not detected:
            return False
    return True

def gen_candidates_from_weight2(weight2_list, include_other_ancillas=False):
    anc_set = {e["ancilla"] for e in weight2_list}
    positions = list(range(0, SHORT_LEN+1))
    if include_other_ancillas:
        anc_list = [0,1,2]
    else:
        anc_list = sorted(list(anc_set)) if anc_set else [0,1,2]
    singles = [(a,q) for a in anc_list for q in positions]
    # all unordered pairs
    pairs = []
    for i in range(len(singles)):
        for j in range(i+1, len(singles)):
            pairs.append((singles[i], singles[j]))
    candidates = [ [s] for s in singles ] + [ [a,b] for (a,b) in pairs ]
    return candidates


# --- Finding optimal detecting regions with minimal CNOTs ---
TRANSFORM_INNER_1 = np.array([1, 1], dtype=np.uint8)            # shape (2,)
TRANSFORM_INNER_2 = np.kron(np.eye(2, dtype=np.uint8), TRANSFORM_INNER_1)  # shape (2,4)
TRANSFORM_INNER_3 = np.kron(np.eye(3, dtype=np.uint8), TRANSFORM_INNER_1)  # shape (3,6)
TRANSFORM_INNER_4 = np.kron(np.eye(4, dtype=np.uint8), TRANSFORM_INNER_1)  # shape (4,8)
TRANSFORM_INNER_5 = np.kron(np.eye(5, dtype=np.uint8), TRANSFORM_INNER_1)  # shape (5,10)
TRANSFORM_INNER = {1: TRANSFORM_INNER_1, 2: TRANSFORM_INNER_2, 3: TRANSFORM_INNER_3, 4: TRANSFORM_INNER_4, 5: TRANSFORM_INNER_5}


def compute_gauge_matrix_index_to_rows(a2as):
    num_split = 0
    tracked = dict()
    index_to_space_time = list()
    for i, (c, n) in enumerate(a2as):
        if c in tracked:
            index_to_space_time.append((tracked[c], c))
            num_split += 1

        if n in tracked:
            index_to_space_time.append((tracked[n], n))
            num_split += 1

        tracked[c] = i
        tracked[n] = i

    inputs = [0,1,2]
    outputs = list(range(3 + 2 * num_split, 3 + 2 * num_split + 3))
    next = 3
    ret = []
    for i, (c, n) in enumerate(a2as):
        rows = [inputs[c], inputs[n], outputs[c], outputs[n]]
        if (i, c) in index_to_space_time:
            rows[2] = next
            inputs[c] = next + 1
            next += 2
        if (i, n) in index_to_space_time:
            rows[3] = next
            inputs[n] = next + 1
            next += 2

        ret.append(rows)
    return num_split, ret


def to_gauge_matrix_and_vector(ops, plist):
    a2as = [(c, n) for (t, c, n) in ops if t == "a2a"]

    n, index_to_rows = compute_gauge_matrix_index_to_rows(a2as)

    A = np.zeros(shape=(6 + 2 * n, 2 * len(a2as)), dtype=np.uint8)

    for i, _ in enumerate(a2as):
        k = 2 * i
        cb, nb, ca, na = index_to_rows[i]
        A[[cb, ca, nb], k] = 1
        A[[cb, ca, na], k + 1] = 1

    marked_cols = set()
    internal_index = []
    k = 0
    for p_index, (type, c_ancilla, n_ancilla) in enumerate(ops):
        internal_index.append(k)
        if type == 'a2a':
            k += 1

    pfaults = [(p['ancilla'], p['position']) for p in plist]
    for p_index, (p_ancilla, p_position) in enumerate(pfaults):
        ii = internal_index[p_position]
        for ii_next, (c_next, n_next) in enumerate(a2as[ii:]):
            c_col_before, n_col_before, _, _ = index_to_rows[ii + ii_next]
            if p_ancilla == c_next:
                if 3 <= c_col_before < 3 + 2 * n:
                    marked_cols.add(c_col_before - 1)
                marked_cols.add(c_col_before)
                break
            if p_ancilla == n_next:
                if 3 <= n_col_before < 3 + 2 * n:
                    marked_cols.add(n_col_before - 1)
                marked_cols.add(n_col_before)
                break
        else:
            cs = [ii_next for ii_next, (c_next, _) in enumerate(a2as) if c_next == p_ancilla]
            ns = [ii_next for ii_next, (_, n_next) in enumerate(a2as) if n_next == p_ancilla]
            if len(cs) and len(ns):
                c, n = cs[-1], ns[-1]
                if c > n:
                    marked_cols.add(index_to_rows[c][2])
                else:
                    marked_cols.add(index_to_rows[n][3])
            if len(cs):
                marked_cols.add(index_to_rows[cs[-1]][2])
            elif len(ns):
                marked_cols.add(index_to_rows[ns[-1]][3])
            else:
                marked_cols.add(p_ancilla)

    return A, marked_cols


def all_bitrows(n):
    if n == 0:
        return np.zeros((1, 0), dtype=np.uint8)
    m = 1 << n
    ints = np.arange(m, dtype=np.uint32)[:, None]    # shape (m,1)
    shifts = np.arange(n, dtype=np.uint32)[None, :]  # shape (1,n)
    return ((ints >> shifts) & 1).astype(np.uint8)   # shape (m,n)


def find_minimal_gauge_solution_vec(A: np.ndarray, marked_cols: set[int], debug=False):
    A = np.asarray(A, dtype=np.uint8)
    m, n_gauges = A.shape
    marked_cols = np.asarray(list(marked_cols), dtype=np.int64)

    B = all_bitrows(n_gauges)
    V = (B @ A.T) & 1
    feasible_mask = V[:, marked_cols].all(axis=1)

    if not feasible_mask.any():
        return None, None, None

    head_w = V[:, :3].sum(axis=1)
    tail_w = V[:, -3:].sum(axis=1)

    inner_mat = V[:, 3:-3]
    k = inner_mat.shape[1] // 2
    T = TRANSFORM_INNER[k]

    if k == 1:
        S = (inner_mat.astype(np.uint8) @ T.astype(np.uint8)) & 1
        inner_w = S.astype(np.int32)
    else:
        S = (inner_mat.astype(np.uint8) @ T.T.astype(np.uint8)) & 1
        inner_w = S.astype(np.int32).sum(axis=1)

    total_weight = head_w + tail_w + inner_w

    INF = 10**9
    total_weight_masked = total_weight.copy()
    total_weight_masked[~feasible_mask] = INF

    best_idx = int(np.argmin(total_weight_masked))
    best_weight = int(total_weight_masked[best_idx])
    best_bits = tuple(int(x) for x in B[best_idx])
    best_v = [int(x) for x in V[best_idx].tolist()]

    return best_bits, best_weight, best_v



# --- main join/process pipeline ---
def join_and_process(forward_dir, backward_dir, out_jsonl_path, gzip_out=True,
                     sample_mod=1):
    recon_iter = mitm_reconstruct_generator(forward_dir, backward_dir)
    # open output
    fh = gzip.open(out_jsonl_path, "wt") if gzip_out else open(out_jsonl_path, "w")
    seq_id = 0
    stats = Counter()
    pattern_counter = Counter()
    optimal_cnot_counter = Counter()
    t0 = time.time()

    pbar = tqdm(total=14398224, desc="joined-seqs", unit="seq")
    for seq_ops in recon_iter:
        # safety check
        if len(seq_ops) != SHORT_LEN: continue
        friendly_ops = [ops[o] for o in seq_ops]

        # compute pattern (per-ancilla target counts)
        tcounts = [0,0,0]
        for op in friendly_ops:
            if op[0] == "d2a":
                tcounts[op[2]] += 1
            else:
                tcounts[op[2]] += 1
        pattern = tuple(sorted(tcounts))
        pattern_counter[pattern] += 1

        # compute weight-2 problematic locations
        w2 = compute_weight2_locations_from_seq(seq_ops)

        # Transform problem to easily solvable description as matrix and vector.
        A, marked_edges = to_gauge_matrix_and_vector(friendly_ops, w2)
        best_bits, best_cost, best_v = find_minimal_gauge_solution_vec(A, marked_edges)

        rec = {
            "seq_id": seq_id,
            "op_indices": seq_ops,                   # exact ops as indices
            "ops": [list(ops[o]) for o in seq_ops],  # readable ops
            "control_pattern": pattern,
            "weight2_problematic": w2,
            "CNOTs_to_FT": best_cost,
            "example_candidate": [best_bits, best_v]
        }
        if (seq_id % sample_mod) == 0:
            fh.write(json.dumps(rec) + "\n")

        seq_id += 1
        stats["total"] += 1
        optimal_cnot_counter[best_cost] += 1

        pbar.update(1)

        # optionally free partials (not implemented per-file here)
    pbar.close()
    fh.close()
    elapsed = time.time() - t0
    # convert pattern_counter keys (tuples) to strings for JSON
    pattern_counts_json = {str(k): v for k, v in pattern_counter.items()}
    optimal_CNOT_counts_json = {str(k) if k is not None else "1 flag not sufficient": v for k, v in optimal_cnot_counter.items()}

    summary = {
        "total_processed": stats["total"],
        "Optimal CNOTs": optimal_CNOT_counts_json,
        "pattern_counts": pattern_counts_json,
        "elapsed_seconds": elapsed
    }
    # save summary
    with open("summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)
    return summary

# --- CLI ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=6, help='Forward depth (default 6)')
    ap.add_argument('--forward-dir', default='mitm_forward')
    ap.add_argument('--backward-dir', default='mitm_backward')
    ap.add_argument('--out', default='full_results.jsonl.gz')
    ap.add_argument('--gzip', action='store_true', help='gzip output')
    ap.add_argument('--sample-mod', type=int, default=1, help='write every M-th record to output')
    ap.add_argument('--max-seqs', type=int, default=None, help='stop after N full sequences (for test)')
    ap.add_argument('--resume', action='store_true', help='reuse existing partials')
    ap.add_argument('--delete-partials', action='store_true', help='delete partial files after joining (saves disk)')
    args = ap.parse_args()

    # compute BFS dist (useful check)
    shortest_len, dist_map = compute_bfs()
    if shortest_len != SHORT_LEN:
        print("Warning: discovered shortest_len:", shortest_len, "expected", SHORT_LEN)

    k = args.k
    if not (0 < k < SHORT_LEN):
        raise ValueError("k must satisfy 0 < k < 11")

    # generate partials (if missing)
    if not os.path.exists(args.forward_dir) or not os.listdir(args.forward_dir):
        print("Generating forward partials (k=%d) ..." % k)
        generate_partials(ZERO, k, args.forward_dir, resume=args.resume)
    else:
        print("Forward partials exist; using", args.forward_dir)
    if not os.path.exists(args.backward_dir) or not os.listdir(args.backward_dir):
        print("Generating backward partials (L-k=%d) ..." % (SHORT_LEN - k))
        generate_partials(H_state, SHORT_LEN - k, args.backward_dir, resume=args.resume)
    else:
        print("Backward partials exist; using", args.backward_dir)

    print("Starting join + process (streaming). Output ->", args.out)
    start = time.time()
    summary = join_and_process(args.forward_dir, args.backward_dir, args.out,
                               gzip_out=args.gzip, sample_mod=args.sample_mod)
    end = time.time()
    print("Finished. Summary:", json.dumps(summary, indent=2))
    print("Total time: {:.2f}s".format(end - start))
