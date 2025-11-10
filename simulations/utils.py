from typing import Callable

import numpy as np
import stim
import json

H_x = H_z = np.array([
    [1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1],
], dtype=np.uint8)
H_x_flag = H_z_flag = np.r_[
    np.eye(1, H_x.shape[1] + 1, dtype=np.uint8),
    np.c_[np.zeros(H_x.shape[0], dtype=np.uint8), H_x]
]
H = np.kron(np.eye(2, dtype=np.uint8), H_x)
H_flag = np.kron(np.eye(2, dtype=np.uint8), H_x_flag)
FLOQ_i = [1, 2, 3, 4]
FLOQ_i_flag = [0, 2, 3, 4, 5]
H_floq_T = H[:, FLOQ_i].T.copy()
H_x_floq_T = H_x[:, FLOQ_i].T.copy()
H_x_flag_floq_T = H_x_flag[:, FLOQ_i_flag].T.copy()

# Specs of Quantinuum H2 (V3.00)
p_1 = 3e-5
p_2 = 1e-3
p_SPAM = 1e-3
p_mem = 1e-4
data_qubits = range(7)

NUM_MAX_CYCLES = 10
NUM_SAMPLES = 10_000_000

syndrome_to_fix = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
], dtype=np.uint8).reshape((2, 2, 2, 7))
flagged_syndrome_to_fix = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
], dtype=np.uint8).reshape((2, 2, 2, 7))

cycles_data = list(range(0, NUM_MAX_CYCLES + 1, 1))
errors_data = 10 ** np.linspace(-2, -4, 21)


def implement_CNOT_circuit(cnots, max_qubit):
    circ = stim.Circuit()
    all_qubits = set(range(max_qubit + 1))
    free_qubits = all_qubits.copy()
    for c, n in cnots:
        if c in free_qubits and n in free_qubits:
            free_qubits -= {c, n}
        else:
            circ.append("Z_ERROR", free_qubits, p_mem)
            free_qubits = all_qubits.copy() - {c, n}
        circ.append("CNOT", [c, n])
        circ.append("DEPOLARIZE2", [c, n], p_2)
    circ.append("Z_ERROR", free_qubits, p_mem)
    return circ


def syndrome_measurement(h1, cnots, h2, base="Z", flags=()):
    max_qubit = max([max(c, n) for c, n in cnots])
    if base == "X":
        h1 = set(range(7, max_qubit + 1)) - set(h1)
        h2 = set(range(7, max_qubit + 1)) - set(h2)
        cnots = [(n, c) for c, n in cnots]
    # print(f"{h1 = }")
    # print(f"{cnots = }")
    # print(f"{h2 = }")

    circ = stim.Circuit()
    for h in h1:
        circ.append("H", h)
    circ.append("Z_ERROR", set(range(7)) - set(h1), p_mem)
    circ += implement_CNOT_circuit(cnots, max_qubit)
    for h in h2:
        circ.append("H", h)
    circ.append("Z_ERROR", set(range(7)) - set(h2), p_mem)

    circ.append("MR", flags, p_SPAM)
    circ.append("MR", range(7, max_qubit + 1 - len(flags)), p_SPAM)
    return circ


def logical_0_state():
    return stim.Circuit(f'''
    H 0 4 6
    Z_ERROR({p_mem}) 1 2 3 5 7
    CX 0 1 4 5 6 3
    DEPOLARIZE2({p_2}) 0 1 4 5 6 3
    Z_ERROR({p_mem}) 2 7
    CX 6 5 4 2 0 3
    DEPOLARIZE2({p_2}) 6 5 4 2 0 3
    Z_ERROR({p_mem}) 1 7
    CX 4 1 3 2
    DEPOLARIZE2({p_2}) 4 1 3 2
    Z_ERROR({p_mem}) 0 5 6 7
    CX 1 7
    DEPOLARIZE2({p_2}) 1 7
    Z_ERROR({p_mem}) 0 2 3 4 5 6
    CX 3 7
    DEPOLARIZE2({p_2}) 3 7
    Z_ERROR({p_mem}) 0 1 2 4 5 6
    CX 5 7
    DEPOLARIZE2({p_2}) 5 7
    Z_ERROR({p_mem}) 0 1 2 3 4 6
    MR({p_SPAM}) 7
    TICK
''')

def perfect_logical_0_state():
    return stim.Circuit(f'''
    H 0 4 6
    CX 0 1 4 5 6 3
    CX 6 5 4 2 0 3
    CX 4 1 3 2
    TICK
''')


perfect_transversal = lambda op: stim.Circuit(f"{op} {' '.join(map(str, data_qubits))}")

perfect_logical_X = perfect_transversal('X')
perfect_logical_Z = perfect_transversal('Z')
perfect_logical_H = perfect_transversal('H')
perfect_logical_M = perfect_transversal('MR')

transversal = lambda op: stim.Circuit(f"""
    {op} {' '.join(map(str, data_qubits))}
    DEPOLARIZE1({p_1}) {' '.join(map(str, data_qubits))}
""")

logical_X = lambda: transversal('X')
logical_Z = lambda: transversal('Z')
logical_S = lambda: transversal('S')
logical_S_DAG = lambda: transversal('S_DAG')
logical_H = lambda: transversal('H')
logical_M = stim.Circuit()
logical_M.append("MR", data_qubits, p_SPAM)


def flatten(zipped_list):
    return list(sum(zipped_list, ()))


def correct(measurements, last_flags=False):
    syndrome = measurements @ H_x.T % 2

    return (measurements ^ np.where(
        last_flags,
        flagged_syndrome_to_fix[tuple(syndrome.T)],
        syndrome_to_fix[tuple(syndrome.T)]
    ).astype(bool))


def get_n_samples(n, circ, post_select_indices) -> tuple[np.ndarray[int, int], int]:
    sampler = circ.compile_sampler()
    ret = np.zeros(shape=(n, sampler.sample(1).shape[1]), dtype=bool)
    k = 0
    N = 0
    phi = 1
    while k < n:
        num_needed = n - k
        batch_size = int(n * phi + 1)
        samples = sampler.sample(shots=batch_size)
        is_good_sample = ~np.any(samples[:, post_select_indices], axis=1)
        good_indices_in_batch = np.where(is_good_sample)[0]
        num_good_found = len(good_indices_in_batch)

        if num_good_found >= num_needed:
            indices_to_use = good_indices_in_batch[:num_needed]
            samples_to_add = samples[indices_to_use]
            last_index_checked = indices_to_use[-1]
            N += last_index_checked + 1
            ret[k:k + num_needed] = samples_to_add
            k += num_needed
        else:
            if num_good_found > 0:
                samples_to_add = samples[good_indices_in_batch]
                ret[k:k + num_good_found] = samples_to_add
            N += batch_size
            k += num_good_found

        if k > 0 and k < n:
            phi = 1.1 * (n - k) * phi / k
        elif k == 0:
            phi = 2 * phi

    return ret, N


def accumulated_error(syndromes_by_cycle):
    num_data_qubits = 7
    num_samples, num_cycles, _, syndromes_per_half_cycle = syndromes_by_cycle.shape

    expected_syndrome_batch = np.zeros((num_samples, 2, syndromes_per_half_cycle), dtype=bool)
    z_errors = np.zeros((num_samples, num_data_qubits), dtype=bool)
    x_errors = np.zeros((num_samples, num_data_qubits), dtype=bool)

    prev_z_flag = np.zeros(num_samples, dtype=bool)

    for t in range(num_cycles):
        measured_syndrome_batch = syndromes_by_cycle[:, t, :, :]
        difference_syndrome_batch = np.logical_xor(expected_syndrome_batch, measured_syndrome_batch)

        x_flag = difference_syndrome_batch[:, 0, 0]
        diff_x_batch = difference_syndrome_batch[:, 0, 1:].astype(np.uint8)

        z_flag = difference_syndrome_batch[:, 1, 0]
        diff_z_batch = difference_syndrome_batch[:, 1, 1:].astype(np.uint8)

        x_fix = syndrome_to_fix[tuple(diff_x_batch.T)]
        flag_x_fix = flagged_syndrome_to_fix[tuple(diff_x_batch.T)]
        x_errors = np.logical_xor(
            x_errors,
            np.where(
                prev_z_flag[:, np.newaxis],
                flag_x_fix,
                np.where(x_flag[:, np.newaxis], False, x_fix)
            )
        )

        z_fix = syndrome_to_fix[tuple(diff_z_batch.T)]
        flag_z_fix = flagged_syndrome_to_fix[tuple(diff_z_batch.T)]
        z_errors = np.logical_xor(
            z_errors,
            np.where(
                x_flag[:, np.newaxis],
                flag_z_fix,
                np.where(z_flag[:, np.newaxis], False, z_fix)
            )
        )

        prev_z_flag = z_flag

        expected_syndrome_batch[:,0,:] = np.where(x_flag[:, np.newaxis], expected_syndrome_batch[:,0,:], measured_syndrome_batch[:,0,:])
        expected_syndrome_batch[:,1,:] = np.where(z_flag[:, np.newaxis], expected_syndrome_batch[:,1,:], measured_syndrome_batch[:,1,:])

    return x_errors, z_errors


def repeated_syndrome_measurement_ket_0(num_cycles: int, measure_syndrome):
    c = perfect_logical_0_state()
    c += measure_syndrome() * num_cycles
    c.append("MR", data_qubits, p_SPAM)
    return c


def repeated_syndrome_measurement_ket_plus(num_cycles, measure_syndrome):
    c = perfect_logical_0_state()
    c += perfect_logical_H
    c += measure_syndrome() * num_cycles
    c += perfect_logical_H
    c.append("MR", data_qubits, p_SPAM)
    return c


def repeated_syndrome_measurement_logical_error_probability(
        num_cycles, num_samples, post_select,
        measure_syndrome,
        preshape_samples: Callable[[np.ndarray, str], tuple[np.ndarray, np.ndarray]],
        last_flag = False):
    ket_0 = repeated_syndrome_measurement_ket_0(num_cycles, measure_syndrome)
    samples_z, N = get_n_samples(num_samples, ket_0, post_select)
    preshaped_samples_z, measurements_z = preshape_samples(samples_z, "X")
    z_sample_x_errors, z_sample_z_errors = accumulated_error(preshaped_samples_z)
    measurements_z ^= z_sample_x_errors
    measurements_z = correct(measurements_z, num_cycles and preshaped_samples_z[:,-1,-1,0,np.newaxis])

    ket_plus = repeated_syndrome_measurement_ket_plus(num_cycles, measure_syndrome)
    samples_x, M = get_n_samples(num_samples, ket_plus, post_select)
    preshaped_samples_x, measurements_x = preshape_samples(samples_x, "Z")
    x_sample_x_errors, x_sample_z_errors = accumulated_error(preshaped_samples_x)
    measurements_x ^= x_sample_z_errors
    measurements_x = correct(measurements_x)

    x_correct = np.sum(measurements_x, axis=1) % 2
    z_correct = np.sum(measurements_z, axis=1) % 2

    return to_data_dict(num_cycles, num_samples, z_correct, x_correct)


def to_data_dict(num_cycles, num_samples, z_logical_measurements, x_logical_measurements):
    z_logical_error_probability = float(np.mean(z_logical_measurements))
    z_standard_error = float(np.std(z_logical_measurements, ddof=1) / np.sqrt(len(z_logical_measurements)))
    z_acceptance_rate = float(len(z_logical_measurements) / num_samples)

    x_logical_error_probability = float(np.mean(x_logical_measurements))
    x_standard_error = float(np.std(x_logical_measurements, ddof=1) / np.sqrt(len(x_logical_measurements)))
    x_acceptance_rate = float(len(x_logical_measurements) / num_samples)

    return {
        "Num Cycles": num_cycles,
        "Num Samples": num_samples,
        "p_1": float(p_1),
        "p_2": float(p_2),
        "p_mem": float(p_mem),
        "p_SPAM": float(p_SPAM),
        "Z Basis": {
            "Logical Error Rate": z_logical_error_probability,
            "Standard Error": z_standard_error,
            "Acceptance Rate": z_acceptance_rate,
        },
        "X Basis": {
            "Logical Error Rate": x_logical_error_probability,
            "Standard Error": x_standard_error,
            "Acceptance Rate": x_acceptance_rate,
        }
    }


def single_raw_SE(qubits, base="Z"):
    return syndrome_measurement(
        [],
        [(q, 7) for q in qubits],
        [],
        base=base
    )

def X_syndrome_measurement(qubits, ancilla):
    all_qubits = set(data_qubits)
    circ = stim.Circuit()
    circ.append("H", ancilla)
    circ.append("DEPOLARIZE1", data_qubits, p_mem)
    for q in qubits:
        circ.append("CNOT", [ancilla, q])
        circ.append("DEPOLARIZE2", [ancilla, q], p_2)
        circ.append("DEPOLARIZE1", all_qubits - {q, ancilla}, p_mem)
    circ.append("DEPOLARIZE1", data_qubits, p_mem)
    circ.append("H", ancilla)
    circ.append("MR", ancilla, p_SPAM)
    return circ


def parallell_raw_SE(base="Z"):
    cnots = [
        (0, 7), (1, 8), (2, 9),
        (1, 7), (2, 8), (3, 9),
        (2, 7), (4, 8), (5, 9),
        (3, 7), (5, 8), (6, 9),
    ]
    return syndrome_measurement([], cnots, [], base=base, flags=[])

def raw_SE():
    return parallell_raw_SE("Z") + parallell_raw_SE("X")

def naive_Z_syndrome_measurement():
    circ = stim.Circuit()
    for i in range(3):
        circ += Z_syndrome_measurement(np.where(H_x[i])[0], 7)
    return circ

def naive_X_syndrome_measurement():
    circ = stim.Circuit()
    for i in range(3):
        circ += X_syndrome_measurement(np.where(H_x[i])[0], 7)
    return circ


def naive_syndrome_measurement():
    circ = stim.Circuit()
    circ += naive_Z_syndrome_measurement()
    circ += naive_X_syndrome_measurement()
    return circ


def combine_results(data, result):
    bases = ("Z Basis", "X Basis")
    if len(data) == 0:
        return {
            k: {k2: [v2] for k2, v2 in v.items()} if k in bases else [v]
            for k, v in result.items()
        }
    for k, v in data.items():
        if k not in ("Z Basis", "X Basis"):
            v.append(result[k])
        else:
            for k2, v2 in v.items():
                v2.append(result[k][k2])
    return data


flag_circuit_syndrome_to_fix = {
    (False, False, False): (),
    (False, False, True): (6,),
    (False, True, False): (4,),
    (False, True, True): (5,),
    (True, False, False): (0,),
    (True, False, True): (3,),
    (True, True, False): (1,),
    (True, True, True): (2,),
}
flag_circuit_syndrome_to_fix_2 = {
    (False, False, False): (),
    (False, False, True): (6,),
    (False, True, False): (2, 3),
    (False, True, True): (5,),
    (True, False, False): (1, 4),
    (True, False, True): (3,),
    (True, True, False): (1,),
    (True, True, True): (2,),
}


LOGICAL_STATE = logical_0_state()
def logical_0_state_RUS(ts):
    ts.do_circuit(LOGICAL_STATE)
    while ts.current_measurement_record()[0]:
        ts = stim.TableauSimulator()
        raise Exception()
        ts.do_circuit(LOGICAL_STATE)
    return ts
