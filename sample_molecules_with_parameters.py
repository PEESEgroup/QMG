import argparse
import logging
import numpy as np
import pandas as pd
from qmg.generator import MoleculeGenerator
from qmg.utils import ConditionalWeightsGenerator
from rdkit import RDLogger
import pickle

RDLogger.DisableLog('rdApp.*')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str,)
    parser.add_argument('--num_heavy_atom', type=int, default=3)
    parser.add_argument('--num_sample', type=int, default=4096)
    parser.add_argument('--smarts', type=str)
    parser.add_argument('--disable_connectivity_position', nargs='+', type=int, default=None)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--no_chemistry_constraint', action='store_true')
    args = parser.parse_args()

    if args.no_chemistry_constraint:
        file_name = f"results_no_chemistry_constraint_bo/{args.task_name}.csv"
    else:
        file_name = f"results_chemistry_constraint_bo/{args.task_name}.csv"
    trial_df = pd.read_csv(file_name)

    cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=args.smarts, disable_connectivity_position=args.disable_connectivity_position)
    if args.smarts:
        random_weight_vector = cwg.generate_conditional_random_weights(random_seed=0)
    else:
        random_weight_vector = np.zeros(cwg.length_all_weight_vector)

    inputs = random_weight_vector
    number_flexible_parameters = len(random_weight_vector[cwg.parameters_indicator == 0.])
    partial_inputs = np.array(trial_df[trial_df["trial_index"] == args.index][[f"x{i+1}" for i in range(number_flexible_parameters)]])[0]
    inputs[cwg.parameters_indicator == 0.] = partial_inputs
    if not args.no_chemistry_constraint:
        inputs = cwg.apply_chemistry_constraint(inputs)

    mg = MoleculeGenerator(args.num_heavy_atom, all_weight_vector=inputs)
    smiles_dict, validity, uniqueness = mg.sample_molecule(args.num_sample)
    print(smiles_dict)
    print("validity:", validity)
    print("uniqueness:", uniqueness)

    if args.no_chemistry_constraint:
        with open(f"results_no_chemistry_constraint_bo/sample_{args.task_name}_{args.index}.pkl", "wb") as f:
            pickle.dump(smiles_dict, f)
    else:
        with open(f"results_chemistry_constraint_bo/sample_{args.task_name}_{args.index}.pkl", "wb") as f:
            pickle.dump(smiles_dict, f)
