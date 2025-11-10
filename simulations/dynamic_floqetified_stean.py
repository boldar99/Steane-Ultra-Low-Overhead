from pprint import pprint

from utils import *
import utils
from tqdm import tqdm


def floq_steane_FT_SE(base="Z"):
    h1 = [10]
    cnots = [
        (3, 8), (5, 9), (6, 7),
        (10, 9),
        (10, 8), (9, 7),
        (7, 8),
        (0, 7), (8, 9),
        (7, 9),
        (10, 7),
        (1, 7), (2, 8), (4, 9),
    ]
    h2 = [10]
    return syndrome_measurement(h1, cnots, h2, base=base, flags=[])

def floq_steane_non_FT_SE(base="Z"):
    cnots = [
        (3, 8), (5, 9), (6, 7),
        (9, 7),
        (7, 8),
        (0, 7), (8, 9),
        (7, 9),
        (1, 7), (2, 8), (4, 9),
    ]
    return syndrome_measurement([], cnots, [], base=base, flags=[])


def apply_corrections(tableao_simulator, x_correct, z_correct):
    for i in x_correct:
        tableao_simulator.x(i)
    for i in z_correct:
        tableao_simulator.z(i)
    return tableao_simulator


FLAGGED_Z_SYNDROME_MEASUREMENT = floq_steane_FT_SE(base="Z")
FLAGGED_X_SYNDROME_MEASUREMENT = floq_steane_FT_SE(base="X")
UNFLAGGED_Z_SYNDROME_MEASUREMENT = floq_steane_non_FT_SE(base="Z")
UNFLAGGED_X_SYNDROME_MEASUREMENT = floq_steane_non_FT_SE(base="X")


flag_circuit_syndrome_to_fix_2 = {
    (False, False, False): (),
    (False, False, True): (6,),
    (False, True, False): (0, 1),
    (False, True, True): (5,),
    (True, False, False): (1, 4),
    (True, False, True): (3,),
    (True, True, False): (1,),
    (True, True, True): (2,),
}

decoder = H_x[:,[1,2,4]].T.copy()


def dynamic_floq_steane_syndrome_measurement(tableao_simulator):

    tableao_simulator.do_circuit(FLAGGED_Z_SYNDROME_MEASUREMENT)
    [z_m1, z_m2, z_m3, flag_z] = tableao_simulator.current_measurement_record()[-4:]
    z_syndromes = (z_m1 ^ z_m2, z_m1 ^ z_m2 ^ z_m3, z_m2)

    if flag_z:
        tableao_simulator.do_circuit(UNFLAGGED_X_SYNDROME_MEASUREMENT)
        [x_m1, x_m2, x_m3] = tableao_simulator.current_measurement_record()[-3:]
        x_syndromes = (x_m1 ^ x_m2, x_m1 ^ x_m2 ^ x_m3, x_m2)
        return apply_corrections(tableao_simulator, [], flag_circuit_syndrome_to_fix_2[x_syndromes])
    elif z_m1 or z_m2 or z_m3:
        tableao_simulator = apply_corrections(tableao_simulator, flag_circuit_syndrome_to_fix[z_syndromes], [])


    tableao_simulator.do_circuit(FLAGGED_X_SYNDROME_MEASUREMENT)
    [x_m1, x_m2, x_m3, flag_x] = tableao_simulator.current_measurement_record()[-4:]
    x_syndromes = (x_m1 ^ x_m2, x_m1 ^ x_m2 ^ x_m3, x_m2)

    if flag_x:
        tableao_simulator.do_circuit(UNFLAGGED_Z_SYNDROME_MEASUREMENT)
        [z_m1, z_m2, z_m3] = tableao_simulator.current_measurement_record()[-3:]
        z_syndromes = (z_m1 ^ z_m2, z_m1 ^ z_m2 ^ z_m3, z_m2)
        return apply_corrections(tableao_simulator, flag_circuit_syndrome_to_fix_2[z_syndromes], [])
    elif x_m1 or x_m2 or x_m3:
        tableao_simulator = apply_corrections(tableao_simulator, [], flag_circuit_syndrome_to_fix[x_syndromes])

    return tableao_simulator


LOGICAL_STATE = perfect_logical_0_state()

def repeated_flagged_parallel_syndrome_measurement(num_cycles, base="Z"):
    ts = stim.TableauSimulator()
    ts.do_circuit(LOGICAL_STATE)
    if base == "X":
        ts.do_circuit(perfect_logical_H)

    for _ in range(num_cycles):
        ts = dynamic_floq_steane_syndrome_measurement(ts)

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


if __name__ == "__main__":
    data_per_cycles = {}
    SAMPLES_PER_CYCLE = NUM_SAMPLES

    for num_cycles in cycles_data:
        z_correct = sample_flagged_parallel_syndrome_measurement(SAMPLES_PER_CYCLE, num_cycles, base="Z")
        x_correct = sample_flagged_parallel_syndrome_measurement(SAMPLES_PER_CYCLE, num_cycles, base="X")

        result = to_data_dict(num_cycles, SAMPLES_PER_CYCLE, z_correct, x_correct)
        data_per_cycles = combine_results(data_per_cycles, result)
        pprint(result)

        with open("data/dynamic_floq_steane_data_by_cycle.json", "w") as outfile:
            json.dump(data_per_cycles, outfile, indent=4)

    data_per_p = {}
    with open("data/dynamic_floq_steane_data_by_p.json", "r") as infile:
        data_per_p = json.load(infile)
    errors_processed = data_per_p["p_2"]

    p_1_to_2_ration = utils.p_1 / utils.p_2
    p_mem_to_2_ration = utils.p_mem / utils.p_2
    for p in errors_data[len(errors_processed):]:
        p_SPAM = p_2 = utils.p_SPAM = utils.p_2 = p
        p_1 = utils.p_1 = p_1_to_2_ration * p
        p_mem = utils.p_mem = p_mem_to_2_ration * p

        utils.LOGICAL_STATE = logical_0_state()
        logical_M = stim.Circuit()
        logical_M.append("MR", data_qubits, p_SPAM)

        FLAGGED_Z_SYNDROME_MEASUREMENT = floq_steane_FT_SE(base="Z")
        FLAGGED_X_SYNDROME_MEASUREMENT = floq_steane_FT_SE(base="X")
        UNFLAGGED_Z_SYNDROME_MEASUREMENT = floq_steane_non_FT_SE(base="Z")
        UNFLAGGED_X_SYNDROME_MEASUREMENT = floq_steane_non_FT_SE(base="X")

        z_correct = sample_flagged_parallel_syndrome_measurement(int(5e-3 * NUM_SAMPLES / p),1, base="Z")
        x_correct = sample_flagged_parallel_syndrome_measurement(int(5e-3 * NUM_SAMPLES / p),1, base="X")

        result = to_data_dict(1, int(5e-3 * NUM_SAMPLES / p), z_correct, x_correct)
        data_per_p = combine_results(data_per_p, result)
        pprint(result)

        with open("data/dynamic_floq_steane_data_by_p.json", "w") as outfile:
            json.dump(data_per_p, outfile, indent=4)
