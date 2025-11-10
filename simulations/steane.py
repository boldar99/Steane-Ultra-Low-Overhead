from pprint import pprint

from utils import *
import utils

def steane_style_measurement(base="Z"):
    h1 = [8, 9, 10, 12, 14]
    cnots = [(8, 7), (12, 11), (10, 13), (12, 13), (9, 11), (10, 7), (8, 11), (9, 10), (14, 8), (14, 10), (14, 12),
             (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13)]
    h2 = [14]
    return syndrome_measurement(h1, cnots, h2, base=base, flags=[14])


def steane_style_syndrome_measurement():
    return steane_style_measurement(base="Z") + steane_style_measurement(base="X")


def preshape_steane_style_measurement_syndromes(samples, _basis="Z"):
    num_data_qubits = 7
    steane_measurements = 7

    num_cycles = (samples.shape[1] - 7) // 16

    filtered_samples = samples

    all_syndromes = filtered_samples[:, :-num_data_qubits]
    measurements = filtered_samples[:, -num_data_qubits:]

    num_samples = filtered_samples.shape[0]

    measurements_by_steane_gadgets = all_syndromes.reshape(num_samples, num_cycles, 2, steane_measurements + 1)

    syndromes_by_steane_gadgets = ((measurements_by_steane_gadgets @ H_x_flag.T) % 2)

    return syndromes_by_steane_gadgets, measurements

if __name__ == "__main__":

    data_per_cycles = {}
    for num_cycles in cycles_data:
        result = repeated_syndrome_measurement_logical_error_probability(
            num_cycles, NUM_SAMPLES,
            list(range(0, num_cycles * 16, 8)),
            steane_style_syndrome_measurement,
            preshape_steane_style_measurement_syndromes,
        )
        data_per_cycles = combine_results(data_per_cycles, result)
        pprint(data_per_cycles)

        with open("data/steane_data_by_cycle.json", "w") as outfile:
            json.dump(data_per_cycles, outfile, indent=4)

    data_per_p = {}
    with open("data/steane_data_by_p.json", "r") as infile:
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
            list(range(0, 16, 8)),
            steane_style_syndrome_measurement,
            preshape_steane_style_measurement_syndromes,
        )
        data_per_p = combine_results(data_per_p, result)
        pprint(data_per_p)

        with open("data/steane_data_by_p.json", "w") as outfile:
            json.dump(data_per_p, outfile, indent=4)
