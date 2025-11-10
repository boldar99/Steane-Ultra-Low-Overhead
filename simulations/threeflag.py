import json
from pprint import pprint

import stim
import numpy as np
import utils
from utils import *
from tqdm import tqdm


def flagged_three_qubit_SE(base="XZZ"):
    h1 = [7]
    cnots = [
        (7, 3), (2, 9), (5, 8),
        (7, 8),
        (7, 0), (3, 9), (4, 8),
        (7, 1), (6, 9), (2, 8),
        (7, 9),
        (7, 2), (5, 9), (1, 8)
    ]
    h2 = [7]
    return syndrome_measurement(
        h1, cnots, h2, flags=[],
        base="Z" if base == "XZZ" else "X"
    )


def correct_shared_flagged(flags, x_syndromes, z_syndromes):
    x_correction = flag_circuit_syndrome_to_fix[x_syndromes]
    z_correction = flag_circuit_syndrome_to_fix[z_syndromes]

    if x_correction == (6,) and flags[1:] == [True, True]:
        x_correction = (1, 2)
    elif z_correction == (6,) and flags != [False, False, True]:
        z_correction = (1, 2)
    elif z_correction == (4,) and flags != [False, True, False]:
        z_correction = (5, 6)

    return x_correction, z_correction


SYNDROME_MEASUREMENT_1 = flagged_three_qubit_SE("XZZ")
SYNDROME_MEASUREMENT_2 = flagged_three_qubit_SE("ZXX")
NAIVE_MEASUREMENT_CIRCUIT = parallell_raw_SE("Z") + parallell_raw_SE("X")


def flagged_parallel_syndrome_measurement(tableao_simulator):
    tableao_simulator.do_circuit(SYNDROME_MEASUREMENT_1)
    flags_1 = tableao_simulator.current_measurement_record()[-3:]
    if np.any(flags_1):
        tableao_simulator.do_circuit(NAIVE_MEASUREMENT_CIRCUIT)
        syndromes = tuple(tableao_simulator.current_measurement_record()[-6:])
        x_correct, z_correct = correct_shared_flagged(
            flags_1,
            syndromes[:3],
            syndromes[3:],
        )
        for i in x_correct:
            tableao_simulator.x(i)
        for i in z_correct:
            tableao_simulator.z(i)
        return tableao_simulator

    tableao_simulator.do_circuit(SYNDROME_MEASUREMENT_2)
    flags_2 = tableao_simulator.current_measurement_record()[-3:]
    if np.any(flags_2):
        tableao_simulator.do_circuit(NAIVE_MEASUREMENT_CIRCUIT)
        syndromes = tuple(tableao_simulator.current_measurement_record()[-6:])
        z_correct, x_correct = correct_shared_flagged(
            flags_2,
            syndromes[3:],
            syndromes[:3],
        )
        for i in x_correct:
            tableao_simulator.x(i)
        for i in z_correct:
            tableao_simulator.z(i)

    return tableao_simulator


LOGICAL_STATE = perfect_logical_0_state()

def repeated_flagged_parallel_syndrome_measurement(num_cycles, base="Z"):
    ts = stim.TableauSimulator()
    ts.do_circuit(LOGICAL_STATE)
    # ts = logical_0_state_RUS(ts)
    if base == "X":
        ts.do_circuit(perfect_logical_H)

    for _ in range(num_cycles):
        ts = flagged_parallel_syndrome_measurement(ts)

    if base == "X":
        ts.do_circuit(perfect_logical_H)
    ts.do_circuit(logical_M)

    measurements = np.array(ts.current_measurement_record()[-7:], dtype=np.int8)
    syndromes = measurements @ H_x.T % 2
    fixed_measurements = (measurements + syndrome_to_fix[syndromes[0], syndromes[1], syndromes[2]]) % 2
    return int(np.sum(fixed_measurements) % 2)


def sample_flagged_parallel_syndrome_measurement(num_samples, num_cycles, base="Z"):
    samples = []
    for _ in tqdm(range(num_samples)):
        samples.append(repeated_flagged_parallel_syndrome_measurement(num_cycles, base))
    return samples


if __name__ == '__main__':
    data_per_cycles = {}
    SAMPLES_PER_CYCLE = NUM_SAMPLES

    for num_cycles in cycles_data:
        z_correct = sample_flagged_parallel_syndrome_measurement(SAMPLES_PER_CYCLE, num_cycles, base="Z")
        x_correct = sample_flagged_parallel_syndrome_measurement(SAMPLES_PER_CYCLE, num_cycles, base="X")

        result = to_data_dict(num_cycles, SAMPLES_PER_CYCLE, z_correct, x_correct)
        data_per_cycles = combine_results(data_per_cycles, result)
        pprint(data_per_cycles)

        with open("data/flagged_three_qubit_data_by_cycle.json", "w") as outfile:
            json.dump(data_per_cycles, outfile, indent=4)

    data_per_p = {}
    with open("data/flagged_three_qubit_data_by_p.json", "r") as infile:
        data_per_p = json.load(infile)
    errors_processed = data_per_p["p_2"]

    p_1_to_2_ration = utils.p_1 / utils.p_2
    p_mem_to_2_ration = utils.p_mem / utils.p_2
    for p in errors_data[len(errors_processed):]:
        p_SPAM = p_2 = utils.p_SPAM = utils.p_2 = p
        p_1 = utils.p_1 = p_1_to_2_ration * p
        p_mem = utils.p_mem = p_mem_to_2_ration * p

        logical_M = stim.Circuit()
        logical_M.append("MR", data_qubits, p_SPAM)

        SYNDROME_MEASUREMENT_1 = flagged_three_qubit_SE("XZZ")
        SYNDROME_MEASUREMENT_2 = flagged_three_qubit_SE("ZXX")
        NAIVE_MEASUREMENT_CIRCUIT = parallell_raw_SE("Z") + parallell_raw_SE("X")

        z_correct = sample_flagged_parallel_syndrome_measurement(int(5e-3 * NUM_SAMPLES / p),1, base="Z")
        x_correct = sample_flagged_parallel_syndrome_measurement(int(5e-3 * NUM_SAMPLES / p),1, base="X")

        result = to_data_dict(1, int(5e-3 * NUM_SAMPLES / p), z_correct, x_correct)
        data_per_p = combine_results(data_per_p, result)
        pprint(data_per_p)

        with open("data/flagged_three_qubit_data_by_p.json", "w") as outfile:
            json.dump(data_per_p, outfile, indent=4)
