import multiprocessing
import pandas as pd
import argparse
from qmg.utils import MoleculeQuantumStateGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
                        default='./dataset/chemical_space/')
    parser.add_argument('--heavy_atom_size', type=int,
                        default=3)
    parser.add_argument('--ncpus', type=int,
                        default=4)
    args = parser.parse_args()

    qg = MoleculeQuantumStateGenerator(heavy_atom_size=args.heavy_atom_size, ncpus=args.ncpus)
    def subfunction_check_state(decimal):
        quantum_state = qg.decimal_to_binary(decimal, qg.n_qubits)
        smiles = qg.ConnectivityToSmiles(*qg.QuantumStateToConnectivity(quantum_state))
        if smiles:
            if not ("." in smiles):
                return smiles
        return 0
    
    index_list = list(range(2**qg.n_qubits))
    with multiprocessing.Pool(processes=args.ncpus) as pool:
        validity_results = pool.map(subfunction_check_state, index_list)

    effective_index = []
    effective_smiles_list = []
    for idx, smiles in zip(index_list, validity_results):
        if smiles:
            effective_index.append(idx)
            effective_smiles_list.append(smiles)
    
    data = pd.DataFrame({"decimal_index": effective_index, "smiles": effective_smiles_list})
    data.to_csv(args.save_dir+f"effective_{args.heavy_atom_size}.csv", index=False)
