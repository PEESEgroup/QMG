import argparse
import logging
import pandas as pd
from qmg.generator import MoleculeGenerator
from rdkit import RDLogger
import pickle
import time

RDLogger.DisableLog('rdApp.*')

def setup_logger(file_name):
    logger = logging.getLogger('MoleculeGeneratorLogger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heavy_atom', type=int, default=3)
    parser.add_argument('--num_sample', type=int, default=4096)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--allow_bond_disconnection', action='store_true')
    parser.add_argument('--no_chemistry_constraint', action='store_true')
    parser.add_argument('--save_data', action='store_true')
    args = parser.parse_args()

    if args.no_chemistry_constraint:
        file_name = f"job_slurm/benchmark_cpu_sample_time/num_{args.num_heavy_atom}_atoms_sample_{args.num_sample}_seed_{args.random_seed}_temperature_{args.temperature}.log"
    else:
        file_name = f"job_slurm/benchmark_cpu_sample_time/num_{args.num_heavy_atom}_atoms_sample_{args.num_sample}_seed_{args.random_seed}_temperature_{args.temperature}.log"
    logger = setup_logger(file_name)
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"random seed of weight vector: {args.random_seed}")
    logger.info(f"temperature parameters: {args.temperature}")

    mg = MoleculeGenerator(args.num_heavy_atom, temperature=args.temperature, 
                           remove_bond_disconnection=not args.allow_bond_disconnection, chemistry_constraint=not args.no_chemistry_constraint)
    start_time = time.time()
    smiles_dict, validity, diversity = mg.sample_molecule(num_sample=args.num_sample, random_seed=args.random_seed)
    end_time = time.time()
    logger.info(f"Time execution of sampling: {end_time - start_time:.6f} seconds.")

    print(smiles_dict)
    # logger.info(smiles_dict)
    logger.info("Validity: {:.2f}%".format(validity * 100))
    logger.info("Diversity: {:.2f}%".format(diversity * 100))
    if args.save_data:
        if args.no_chemistry_constraint:
            with open(f"results_without_constraint/heavy_atom_{args.num_heavy_atom}_seed_{args.random_seed}.pkl", "wb") as f:
                pickle.dump(smiles_dict, f)
        else:
            with open(f"results/heavy_atom_{args.num_heavy_atom}_seed_{args.random_seed}.pkl", "wb") as f:
                pickle.dump(smiles_dict, f)
