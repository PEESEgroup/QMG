"""
Multi-objective Bayesian optimization for conditional quantum generator (further optimization).
"""

from qmg.generator import MoleculeGenerator
from qmg.utils import ConditionalWeightsGenerator, FitnessCalculatorWrapper
from rdkit import RDLogger
import numpy as np
import pandas as pd

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax import SearchSpace, ParameterType, RangeParameter
from ax.core.observation import ObservationFeatures
from ax.core.arm import Arm
import torch
import argparse
import logging

torch.set_default_dtype(torch.float64)
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
    parser.add_argument('--task_name', type=str,)
    parser.add_argument('--iteration_number', type=int, default=0)
    parser.add_argument('--task', nargs='+', type=str, default=None,
                        choices=["validity", "uniqueness", "qed", "logP", "tpsa", "sascore", "SAscore", "ClogP", "CMR",
                                 "product_validity_uniqueness", "product_uniqueness_validity"])
    parser.add_argument('--condition', nargs='+', type=str, default=None) # can be None or float number
    parser.add_argument('--objective', nargs='+', type=str, choices=["minimize", "maximize"], default=None)
    parser.add_argument('--num_heavy_atom', type=int, default=5)
    parser.add_argument('--num_sample', type=int, default=10000)
    parser.add_argument('--smarts', type=str)
    parser.add_argument('--disable_connectivity_position', nargs='+', type=int, default=None)
    parser.add_argument('--no_chemistry_constraint', action='store_true')
    parser.add_argument('--num_iterations', type=int)
    args = parser.parse_args()

    assert len(args.task) == len(args.condition) == len(args.objective)
    if args.no_chemistry_constraint:
        data_dir = "results_no_chemistry_constraint_bo"
    else:
        data_dir = "results_chemistry_constraint_bo"
    file_name = f"{data_dir}/{args.task_name}.log"

    previous_csv_path = f"{data_dir}/{args.task_name}_{args.iteration_number}.csv"
    previous_data = pd.read_csv(previous_csv_path)
    logger = setup_logger(file_name)
    logger.info(f"*** Further optimization based on the previous results ***")
    logger.info(f"Previous data path: {previous_csv_path}")
    logger.info(f"Task name: {args.task_name}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Condition: {args.condition}")
    logger.info(f"objective: {args.objective}")
    logger.info(f"# of heavy atoms: {args.num_heavy_atom}")
    logger.info(f"# of samples: {args.num_sample}")
    logger.info(f"smarts: {args.smarts}")
    logger.info(f"disable_connectivity_position: {args.disable_connectivity_position}")
    logger.info(f"Using cuda: {torch.cuda.is_available()}")

    cwg = ConditionalWeightsGenerator(args.num_heavy_atom, smarts=args.smarts, disable_connectivity_position=args.disable_connectivity_position)
    if args.smarts:
        random_weight_vector = cwg.generate_conditional_random_weights(random_seed=0)
    else:
        random_weight_vector = np.zeros(cwg.length_all_weight_vector)

    number_flexible_parameters = len(random_weight_vector[cwg.parameters_indicator == 0.])
    logger.info(f"Number of flexible parameters: {number_flexible_parameters}")
    random_weight_vector[cwg.parameters_indicator == 0.] = np.random.rand(len(random_weight_vector[cwg.parameters_indicator == 0.]))

    fitness_calculator = FitnessCalculatorWrapper(task_list=args.task, condition=args.condition)

    ################################### Generation Strategy ###################################
    model_dict = {'MOO': Models.MOO, 'GPEI': Models.GPEI, 'SAASBO': Models.SAASBO,}
    gs = GenerationStrategy(
        steps=[
    #         only use this when there is no initial data
            GenerationStep(
                model=model_dict['GPEI'],
                num_trials=-1,  # No limitation on how many trials should be produced from this step
                max_parallelism=1,  # Parallelism limit for this step, often lower than for Sobol
                model_kwargs = {"torch_dtype": torch.float64, "torch_device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                                },
            ),
        ]
    )
    ax_client = AxClient(random_seed = 42, generation_strategy = gs) # set the random seed for BO for reproducibility
    ax_client.create_experiment(
        name=args.task_name,
        parameters=[
            {
                "name": f"x{i+1}",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float"
            }
            for i in range(number_flexible_parameters)
        ],
        objectives={task: ObjectiveProperties(minimize = objective=="minimize",) for task, objective in zip(args.task, args.objective)},
        overwrite_existing_experiment=True,
        is_test=True,
    )
    ################################### Load previous data ###################################
    for i, row in previous_data.iterrows():
        input_parameters = row[[f"x{i+1}" for i in range(number_flexible_parameters)]].to_dict()
        output_response = row[args.task].to_dict()
        ax_client.attach_trial(input_parameters)
        ax_client.complete_trial(i, output_response)
    ##########################################################################################

    def evaluate(parameters):
        partial_inputs = np.array([parameters.get(f"x{i+1}") for i in range(number_flexible_parameters)])
        if args.smarts:
            inputs = random_weight_vector
            inputs[cwg.parameters_indicator == 0.] = partial_inputs
        else:
            inputs = partial_inputs
        if not args.no_chemistry_constraint:
            inputs = cwg.apply_chemistry_constraint(inputs)
        mg = MoleculeGenerator(args.num_heavy_atom, all_weight_vector=inputs)
        smiles_dict, validity, uniqueness = mg.sample_molecule(args.num_sample)
        score_dict, score_pure_dict = fitness_calculator.evaluate(smiles_dict)
        for task, objective, condition in zip(args.task, args.objective, args.condition):
            if str(condition) == "None":
                logger.info(f"{task} ({objective}): {score_dict[task][0]:.3f}")
            else:
                logger.info(f"{task} (close to {condition}): {score_pure_dict[task]:.3f}")
        # Set standard error to None if the noise level is unknown.
        return score_dict

    for i in range(args.num_iterations):
        parameters, trial_index = ax_client.get_next_trial()
        logger.info(f"Iteration number: {trial_index}")
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        trial_df = ax_client.get_trials_data_frame()
        trial_df.to_csv(f"{data_dir}/{args.task_name}_{args.iteration_number+1}.csv", index=False)