"""
Export connectome connectivity matrix to SS_RNN config file format.

This script converts the connectome data into a JSON config file that can be used
with the SS_RNN network simulation framework.
"""

import pickle
import pandas as pd
import numpy as np
import json
from scipy.sparse import load_npz, spmatrix
from collections import Counter

def load_connectivity(path: str, name: str) -> tuple[spmatrix, dict]:
    sparse_matrix = load_npz(f'{path}/{name}_matrix.npz')
    with open(f'{path}/{name}_mapping.pkl', 'rb') as fp:
        mapping = pickle.load(fp)
    return sparse_matrix, mapping

def get_cell_type_type(cell_type, e_neuron_order, i_neuron_order):
    """Determine if cell type is E (excitatory) or I (inhibitory)"""
    if cell_type in i_neuron_order:
        return 'I'
    elif cell_type in e_neuron_order:
        return 'E'
    else:
        # Default to E if unknown
        return 'E'

print("=== EXPORTING CONNECTOME TO SS_RNN CONFIG ===\n")

# Load connectome data
CONNECTIVITY_DIR = '.'
syn_mat_sparse, mapping = load_connectivity(CONNECTIVITY_DIR, 'network_synapses')
neurons_df = pd.read_csv('connectome_neurons.csv')

print(f"Loaded {len(neurons_df)} neurons")
print(f"Connectivity matrix shape: {syn_mat_sparse.shape}")

# Create mappings
root_id_to_cell_type = dict(zip(neurons_df['root_id'], neurons_df['cell_type']))
matrix_idx_to_cell_type = {}
for idx, root_id in mapping.items():
    matrix_idx_to_cell_type[idx] = root_id_to_cell_type.get(root_id, 'Unknown')

# Get cell type counts and ordering
cell_type_counts = Counter(matrix_idx_to_cell_type.values())
e_neuron_order = ['23P', '4P', '5P-IT', '5P-PT', '5P-NP', '6P-IT', '6P-CT', '6P-U', 'WM-P', 'Unsure E']
i_neuron_order = ['BC', 'MC', 'BPC', 'NGC', 'Unsure I']
ordered_cell_types = [ct for ct in e_neuron_order + i_neuron_order if ct in cell_type_counts]

print(f"\nFound {len(ordered_cell_types)} cell types:")
for ct in ordered_cell_types:
    print(f"  {ct}: {cell_type_counts[ct]} neurons ({get_cell_type_type(ct, e_neuron_order, i_neuron_order)})")

# Compute connectivity statistics
n_cell_types = len(ordered_cell_types)
cell_type_to_idx = {ct: i for i, ct in enumerate(ordered_cell_types)}

# Matrices for connection counts and synapse totals
connection_counts = np.zeros((n_cell_types, n_cell_types), dtype=int)
synapse_totals = np.zeros((n_cell_types, n_cell_types))
connection_probabilities = np.zeros((n_cell_types, n_cell_types))
mean_synapses_per_connection = np.zeros((n_cell_types, n_cell_types))

print("\n=== COMPUTING CONNECTIVITY STATISTICS ===")
for i in range(syn_mat_sparse.shape[0]):
    source_cell_type = matrix_idx_to_cell_type.get(i, 'Unknown')
    if source_cell_type not in cell_type_to_idx:
        continue
    source_idx = cell_type_to_idx[source_cell_type]
    
    # Get outgoing connections
    row = syn_mat_sparse[i, :]
    if row.nnz > 0:
        targets = row.indices
        weights = row.data
        for target_idx, weight in zip(targets, weights):
            target_cell_type = matrix_idx_to_cell_type.get(target_idx, 'Unknown')
            if target_cell_type not in cell_type_to_idx:
                continue
            target_type_idx = cell_type_to_idx[target_cell_type]
            
            connection_counts[source_idx, target_type_idx] += 1
            synapse_totals[source_idx, target_type_idx] += weight

# Compute probabilities and mean synapses
for i, source_type in enumerate(ordered_cell_types):
    for j, target_type in enumerate(ordered_cell_types):
        n_connections = connection_counts[i, j]
        source_count = cell_type_counts[source_type]
        target_count = cell_type_counts[target_type]
        
        # Compute connection probability
        if i == j:
            # Self-connections: exclude self-connections
            possible_connections = source_count * (target_count - 1)
        else:
            possible_connections = source_count * target_count
        
        if possible_connections > 0:
            connection_probabilities[i, j] = n_connections / possible_connections
        
        # Compute mean synapses per connection
        if n_connections > 0:
            mean_synapses_per_connection[i, j] = synapse_totals[i, j] / n_connections

# Create SS_RNN config structure
config = {
    "name": "connectome_based_network",
    "description": "Network based on connectome connectivity data",
    "num_stimuli": 20,
    "num_epochs": 5,
    "stim_duration": 250,
    "break_time": 50,
    "start_time": 100,
    "base_weight_scale": 1.0,
    "weight_scale_method": "sqrt_k",
    "inhibitory_weight_scale": 25.0,
    "input_amplitude_scale": 1.0,
    "stimulus_distribution": "dirichlet",
    "stimulus_alpha": 0.5,
    "scale_factor": 1000.0,
    "simulation_params": {
        "tau_m_ms": 20.0,
        "tau_syn_ms": 1.0,
        "R_m_Mohm": 100.0,
        "v_rest_mV": -65.0,
        "v_thresh_mV": -55.0,
        "v_reset_mV": -65.0,
        "refractory_ms": 2.0
    },
    "input_layer": {
        "n": 200,
        "name": "Layer0"
    },
    "populations": [],
    "connections": []
}

# Add populations
print("\n=== ADDING POPULATIONS ===")
for cell_type in ordered_cell_types:
    n = cell_type_counts[cell_type]
    cell_type_type = get_cell_type_type(cell_type, e_neuron_order, i_neuron_order)
    config["populations"].append({
        "name": cell_type,
        "n": int(n),
        "type": cell_type_type
    })
    print(f"  {cell_type}: {n} neurons ({cell_type_type})")

# Add connections based on connectome data
print("\n=== ADDING CONNECTIONS ===")
connection_threshold = 0.001  # Minimum connection probability to include

for i, source_type in enumerate(ordered_cell_types):
    for j, target_type in enumerate(ordered_cell_types):
        p = connection_probabilities[i, j]
        mean_synapses = mean_synapses_per_connection[i, j]
        
        # Only include connections above threshold
        if p > connection_threshold:
            # Use mean synapses as base weight (normalized)
            # Convert to lognormal distribution parameters
            # For lognormal: if mean = exp(mu + sigma^2/2), we can estimate mu and sigma
            if mean_synapses > 0:
                # Estimate lognormal parameters from mean
                # Using a reasonable sigma and solving for mu
                sigma = 0.5  # Standard deviation in log space
                mu = np.log(mean_synapses) - (sigma**2) / 2
                mu = max(mu, -2.0)  # Cap mu to reasonable values
            else:
                mu = 0.0
                sigma = 0.5
            
            connection_config = {
                "source": source_type,
                "target": target_type,
                "p": float(p),
                "weight_dist": {
                    "dist": "lognormal",
                    "mu": float(mu),
                    "sigma": float(sigma)
                }
            }
            config["connections"].append(connection_config)
            print(f"  {source_type} -> {target_type}: p={p:.4f}, mean_synapses={mean_synapses:.2f}")

# Save config file
output_file = 'connectome_ssrnn_config.json'
with open(output_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n=== CONFIG FILE SAVED ===")
print(f"Saved to: {output_file}")
print(f"Total populations: {len(config['populations'])}")
print(f"Total connections: {len(config['connections'])}")

# Also save a summary statistics file
stats = {
    "cell_types": ordered_cell_types,
    "cell_type_counts": {ct: int(cell_type_counts[ct]) for ct in ordered_cell_types},
    "connection_probabilities": {
        f"{ordered_cell_types[i]}_{ordered_cell_types[j]}": float(connection_probabilities[i, j])
        for i in range(n_cell_types) for j in range(n_cell_types)
        if connection_probabilities[i, j] > connection_threshold
    },
    "mean_synapses_per_connection": {
        f"{ordered_cell_types[i]}_{ordered_cell_types[j]}": float(mean_synapses_per_connection[i, j])
        for i in range(n_cell_types) for j in range(n_cell_types)
        if connection_probabilities[i, j] > connection_threshold
    }
}

stats_file = 'connectome_ssrnn_stats.json'
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"Statistics saved to: {stats_file}")


