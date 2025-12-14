import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle

# Load data
df = pd.read_csv('connectome_neurons.csv')
syn_mat = load_npz('network_synapses_matrix.npz')
mapping = pickle.load(open('network_synapses_mapping.pkl', 'rb'))

# Create mapping from root_id to matrix index
root_id_to_idx = {root_id: idx for idx, root_id in mapping.items()}
df['matrix_idx'] = df['root_id'].map(root_id_to_idx)
df_with_idx = df[df['matrix_idx'].notna()].copy()

# Compute matrix statistics
def get_matrix_stats(idx):
    if pd.isna(idx):
        return 0, 0, 0, 0
    idx = int(idx)
    out_deg = syn_mat[idx, :].nnz
    in_deg = syn_mat[:, idx].nnz
    out_syn = syn_mat[idx, :].data.sum() if out_deg > 0 else 0
    in_syn = syn_mat[:, idx].data.sum() if in_deg > 0 else 0
    return out_deg, in_deg, out_syn, in_syn

df_with_idx['matrix_out_deg'] = df_with_idx['matrix_idx'].apply(lambda x: get_matrix_stats(x)[0])
df_with_idx['matrix_in_deg'] = df_with_idx['matrix_idx'].apply(lambda x: get_matrix_stats(x)[1])
df_with_idx['matrix_out_syn'] = df_with_idx['matrix_idx'].apply(lambda x: get_matrix_stats(x)[2])
df_with_idx['matrix_in_syn'] = df_with_idx['matrix_idx'].apply(lambda x: get_matrix_stats(x)[3])

# Compare
print('=== SYNAPSE COUNT COMPARISON ===')
print(f'\nTotal neurons in CSV: {len(df)}')
print(f'Neurons in matrix: {len(df_with_idx)}')

print(f'\nIncoming synapses:')
print(f'  CSV total: {df_with_idx["num_of_incoming_synapses"].sum():,.0f}')
print(f'  Matrix total: {df_with_idx["matrix_in_syn"].sum():,.0f}')
print(f'  Difference: {df_with_idx["num_of_incoming_synapses"].sum() - df_with_idx["matrix_in_syn"].sum():,.0f}')

print(f'\nOutgoing synapses:')
print(f'  CSV total: {df_with_idx["num_of_outgoing_synapses"].sum():,.0f}')
print(f'  Matrix total: {df_with_idx["matrix_out_syn"].sum():,.0f}')
print(f'  Difference: {df_with_idx["num_of_outgoing_synapses"].sum() - df_with_idx["matrix_out_syn"].sum():,.0f}')

print(f'\nPer-neuron differences (incoming):')
df_with_idx['in_diff'] = df_with_idx['num_of_incoming_synapses'] - df_with_idx['matrix_in_syn']
print(f'  Mean: {df_with_idx["in_diff"].mean():.2f}')
print(f'  Median: {df_with_idx["in_diff"].median():.2f}')
print(f'  Max: {df_with_idx["in_diff"].max():.2f}')
print(f'  Min: {df_with_idx["in_diff"].min():.2f}')

print(f'\nNeurons with largest differences (missing synapses):')
print(df_with_idx.nlargest(10, 'in_diff')[['root_id', 'cell_type', 'num_of_incoming_synapses', 'matrix_in_syn', 'in_diff']].to_string())

print(f'\nChecking for neurons with synapses but no connections in matrix:')
no_connections = df_with_idx[df_with_idx['matrix_in_deg'] == 0]
print(f'  Neurons with 0 incoming connections in matrix: {len(no_connections)}')
if len(no_connections) > 0:
    print(f'  But have incoming synapses in CSV: {no_connections[no_connections["num_of_incoming_synapses"] > 0].shape[0]}')
    print(f'\nSample:')
    print(no_connections[no_connections['num_of_incoming_synapses'] > 0].head()[['root_id', 'cell_type', 'num_of_incoming_synapses', 'matrix_in_deg', 'matrix_in_syn']].to_string())

print(f'\n=== CHECKING FOR EXTERNAL INPUTS ===')
print(f'\nComparing ds_ (downsampled?) vs num_of_ columns:')
print(f'  ds_num_of_incoming_synapses total: {df["ds_num_of_incoming_synapses"].sum():,.0f}')
print(f'  num_of_incoming_synapses total: {df["num_of_incoming_synapses"].sum():,.0f}')
print(f'  Difference: {df["ds_num_of_incoming_synapses"].sum() - df["num_of_incoming_synapses"].sum():,.0f}')

print(f'\nChecking if matrix matches ds_ or num_of_ columns:')
df_with_idx['ds_in_diff'] = df_with_idx['ds_num_of_incoming_synapses'] - df_with_idx['matrix_in_syn']
print(f'  Mean difference (ds vs matrix): {df_with_idx["ds_in_diff"].mean():.2f}')
print(f'  Mean difference (num_of vs matrix): {df_with_idx["in_diff"].mean():.2f}')

print(f'\nNeurons where ds_ count > num_of_ count (possible external inputs?):')
external_candidates = df_with_idx[df_with_idx['ds_num_of_incoming_synapses'] > df_with_idx['num_of_incoming_synapses']]
print(f'  Count: {len(external_candidates)}')
if len(external_candidates) > 0:
    print(f'  Total extra synapses: {(external_candidates["ds_num_of_incoming_synapses"] - external_candidates["num_of_incoming_synapses"]).sum():,.0f}')
    print(f'\nSample:')
    print(external_candidates.head()[['root_id', 'cell_type', 'ds_num_of_incoming_synapses', 'num_of_incoming_synapses', 'matrix_in_syn']].to_string())

