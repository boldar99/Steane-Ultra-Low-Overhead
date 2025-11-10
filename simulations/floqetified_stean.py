from pprint import pprint

from utils import *
import utils


floq_steane_ancilla_qubits = range(7, 11)

def floq_steane_style_measurement(base="Z"):
    h1 = [8, 11]
    cnots = [
        (0, 7), (6, 9), (5, 10), (11, 10), (10, 9), (8, 10), (9, 7),
        (7, 10), (8, 9), (11, 7), (11, 9), (1, 7), (2, 8), (3, 9), (4, 10)
    ]
    h2 = [11]
    return syndrome_measurement(h1, cnots, h2, base=base, flags=[11])


def floq_steane_style_syndrome_measurement():
    return floq_steane_style_measurement(base="Z") + floq_steane_style_measurement(base="X")


def preshape_floq_steane_style_measurement_syndromes(samples, basis):
    num_data_qubits = 7
    floq_steane_measurements = 4

    num_cycles = (samples.shape[1] - 7) // (2 * floq_steane_measurements + 2)

    filtered_samples = samples

    num_samples = filtered_samples.shape[0]
    all_syndromes = filtered_samples[:, :-num_data_qubits]
    measurements = filtered_samples[:, -num_data_qubits:]

    measurements_by_floq_steane_gadgets = all_syndromes.reshape(num_samples, num_cycles, 2, floq_steane_measurements + 1).astype(np.uint8)

    syndromes_by_floq_steane_gadgets = (measurements_by_floq_steane_gadgets @ H_x_flag_floq_T) % 2

    return syndromes_by_floq_steane_gadgets, measurements


if __name__ == "__main__":

    data_per_cycles = {}
    for num_cycles in cycles_data:
        result = repeated_syndrome_measurement_logical_error_probability(
            num_cycles, NUM_SAMPLES, [],
            floq_steane_style_syndrome_measurement,
            preshape_floq_steane_style_measurement_syndromes
        )
        data_per_cycles = combine_results(data_per_cycles, result)
        pprint(data_per_cycles)

        with open("data/floquetified_steane_data_by_cycle.json", "w") as outfile:
            json.dump(data_per_cycles, outfile, indent=4)

    data_per_p = {}
    with open("data/floquetified_steane_data_by_p.json", "r") as infile:
        data_per_p = json.load(infile)
    errors_processed = data_per_p["p_2"]

    p_1_to_2_ration = utils.p_1 / utils.p_2
    p_mem_to_2_ration = utils.p_mem / utils.p_2
    for p in errors_data[len(errors_processed):]:
        p_SPAM = p_2 = utils.p_SPAM = utils.p_2 = p
        p_1 = utils.p_1 = p_1_to_2_ration * p
        p_mem = utils.p_mem = p_mem_to_2_ration * p
        result = repeated_syndrome_measurement_logical_error_probability(
            1,
            int(5e-3 * NUM_SAMPLES / p),
            [],
            floq_steane_style_syndrome_measurement,
            preshape_floq_steane_style_measurement_syndromes,
            last_flag=True,
        )
        data_per_p = combine_results(data_per_p, result)
        pprint(result)

        with open("data/floquetified_steane_data_by_p.json", "w") as outfile:
            json.dump(data_per_p, outfile, indent=4)
