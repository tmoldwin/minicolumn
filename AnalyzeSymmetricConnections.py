import pickle
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, spmatrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_connectivity(path: str, name: str) -> tuple[spmatrix, dict]:
    sparse_matrix = load_npz(f'{path}/{name}_matrix.npz')
    with open(f'{path}/{name}_mapping.pkl', 'rb') as fp:
        mapping = pickle.load(fp)
    return sparse_matrix, mapping

def get_cell_type_color(cell_type, e_neuron_order, i_neuron_order):
    """E=red, I=blue"""
    if cell_type in i_neuron_order:
        return '#3498db'  # Blue for inhibitory
    else:
        return '#e74c3c'  # Red for excitatory

print("Loading data...")
CONNECTIVITY_DIR = '.'
syn_mat_sparse, mapping = load_connectivity(CONNECTIVITY_DIR, 'network_synapses')

neurons_df = pd.read_csv('connectome_neurons.csv')
print(f"Loaded {len(neurons_df)} neurons")

# Create mappings
root_id_to_cell_type = dict(zip(neurons_df['root_id'], neurons_df['cell_type']))
root_id_to_clf_type = dict(zip(neurons_df['root_id'], neurons_df['clf_type']))

matrix_idx_to_cell_type = {}
matrix_idx_to_clf_type = {}
for idx, root_id in mapping.items():
    matrix_idx_to_cell_type[idx] = root_id_to_cell_type.get(root_id, 'Unknown')
    matrix_idx_to_clf_type[idx] = root_id_to_clf_type.get(root_id, 'Unknown')

# Get cell type counts and ordering
cell_type_counts = Counter(matrix_idx_to_cell_type.values())
unique_cell_types = sorted(set(matrix_idx_to_cell_type.values()))
e_neuron_order = ['23P', '4P', '5P-IT', '5P-PT', '5P-NP', '6P-IT', '6P-CT', '6P-U', 'WM-P', 'Unsure E']
i_neuron_order = ['BC', 'MC', 'BPC', 'NGC', 'Unsure I']
ordered_cell_types = [ct for ct in e_neuron_order + i_neuron_order if ct in cell_type_counts]

# Create figure directory
figure_dir = 'figures/symmetric_connections'
os.makedirs(figure_dir, exist_ok=True)

print("\n=== ANALYZING SYMMETRIC CONNECTIONS ===")

# Initialize matrices
n_cell_types = len(unique_cell_types)
cell_type_to_idx = {ct: i for i, ct in enumerate(unique_cell_types)}

# Matrices for connection fractions (real data)
forward_only_fraction_matrix = np.zeros((n_cell_types, n_cell_types))  # Fraction of forward-only connections
reverse_only_fraction_matrix = np.zeros((n_cell_types, n_cell_types))  # Fraction of reverse-only connections
symmetric_fraction_matrix = np.zeros((n_cell_types, n_cell_types))  # Fraction of symmetric connections

# Matrices for null hypothesis fractions
forward_only_null_matrix = np.zeros((n_cell_types, n_cell_types))
reverse_only_null_matrix = np.zeros((n_cell_types, n_cell_types))
symmetric_null_matrix = np.zeros((n_cell_types, n_cell_types))

# For each cell type pair, analyze connection fractions
print("\nComputing connection fractions for each cell type pair...")

for i, source_type in enumerate(unique_cell_types):
    for j, target_type in enumerate(unique_cell_types):
        # Get all neurons of each type
        source_neurons = [idx for idx, ct in matrix_idx_to_cell_type.items() if ct == source_type]
        target_neurons = [idx for idx, ct in matrix_idx_to_cell_type.items() if ct == target_type]
        
        # Count unique neuron pairs by connection type
        forward_only_pairs = 0  # A→B exists but B→A doesn't
        reverse_only_pairs = 0  # B→A exists but A→B doesn't
        symmetric_pairs = 0  # Both A→B and B→A exist
        
        # Check each possible neuron pair
        for source_idx in source_neurons:
            for target_idx in target_neurons:
                # Check if source→target exists
                source_row = syn_mat_sparse[source_idx, :]
                has_forward = False
                if source_row.nnz > 0 and target_idx in source_row.indices:
                    has_forward = True
                
                # Check if target→source exists
                target_row = syn_mat_sparse[target_idx, :]
                has_reverse = False
                if target_row.nnz > 0 and source_idx in target_row.indices:
                    has_reverse = True
                
                # Categorize the pair
                if has_forward and has_reverse:
                    symmetric_pairs += 1
                elif has_forward:
                    forward_only_pairs += 1
                elif has_reverse:
                    reverse_only_pairs += 1
        
        # Calculate total pairs with any connection
        total_pairs = forward_only_pairs + reverse_only_pairs + symmetric_pairs
        total_possible_pairs = len(source_neurons) * len(target_neurons)
        
        # Calculate actual fractions (normalize to sum to 1)
        if total_pairs > 0:
            forward_only_fraction_matrix[i, j] = forward_only_pairs / total_pairs
            reverse_only_fraction_matrix[i, j] = reverse_only_pairs / total_pairs
            symmetric_fraction_matrix[i, j] = symmetric_pairs / total_pairs
        else:
            forward_only_fraction_matrix[i, j] = 0
            reverse_only_fraction_matrix[i, j] = 0
            symmetric_fraction_matrix[i, j] = 0
        
        # Calculate connection probabilities for null hypothesis
        # p = probability of A->B, q = probability of B->A
        if total_possible_pairs > 0:
            p = (forward_only_pairs + symmetric_pairs) / total_possible_pairs
            q = (reverse_only_pairs + symmetric_pairs) / total_possible_pairs
        else:
            p = 0
            q = 0
        
        # Calculate null hypothesis fractions (assuming independent connections)
        # Under null: P(A->B only) = p*(1-q), P(B->A only) = (1-p)*q, P(both) = p*q
        # Normalize to only pairs with at least one connection: total = p + q - p*q
        if p > 0 or q > 0:
            total_null = p + q - p*q
            if total_null > 0:
                forward_only_null_matrix[i, j] = p * (1 - q) / total_null
                reverse_only_null_matrix[i, j] = (1 - p) * q / total_null
                symmetric_null_matrix[i, j] = p * q / total_null
            else:
                forward_only_null_matrix[i, j] = 0
                reverse_only_null_matrix[i, j] = 0
                symmetric_null_matrix[i, j] = 0
        else:
            forward_only_null_matrix[i, j] = 0
            reverse_only_null_matrix[i, j] = 0
            symmetric_null_matrix[i, j] = 0

# Reorder matrices to match ordered_cell_types
ordered_to_original_idx = {ct: unique_cell_types.index(ct) for ct in ordered_cell_types}

forward_only_fraction_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
reverse_only_fraction_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
symmetric_fraction_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))

forward_only_null_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
reverse_only_null_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
symmetric_null_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))

for i, ct_i in enumerate(ordered_cell_types):
    orig_i = ordered_to_original_idx[ct_i]
    for j, ct_j in enumerate(ordered_cell_types):
        orig_j = ordered_to_original_idx[ct_j]
        forward_only_fraction_ordered[i, j] = forward_only_fraction_matrix[orig_i, orig_j]
        reverse_only_fraction_ordered[i, j] = reverse_only_fraction_matrix[orig_i, orig_j]
        symmetric_fraction_ordered[i, j] = symmetric_fraction_matrix[orig_i, orig_j]
        forward_only_null_ordered[i, j] = forward_only_null_matrix[orig_i, orig_j]
        reverse_only_null_ordered[i, j] = reverse_only_null_matrix[orig_i, orig_j]
        symmetric_null_ordered[i, j] = symmetric_null_matrix[orig_i, orig_j]

# Print statistics
print("\n=== CONNECTION FRACTION STATISTICS ===")
print(f"{'Source':<20} {'Target':<20} {'Forward-Only':<15} {'Reverse-Only':<15} {'Symmetric':<15} {'Sum':<15}")
print("-" * 100)

for i, source_type in enumerate(ordered_cell_types):
    for j, target_type in enumerate(ordered_cell_types):
        forward_only = forward_only_fraction_ordered[i, j]
        reverse_only = reverse_only_fraction_ordered[i, j]
        symmetric = symmetric_fraction_ordered[i, j]
        total = forward_only + reverse_only + symmetric
        
        if total > 0:
            print(f"{source_type:<20} {target_type:<20} {forward_only:<15.4f} {reverse_only:<15.4f} {symmetric:<15.4f} {total:<15.4f}")

# Create visualizations
print("\n=== CREATING VISUALIZATIONS ===")
sns.set_style("whitegrid")

# Single figure with all symmetry analyses: 2 rows x 3 columns
# Top row: Null hypothesis (independent connections)
# Bottom row: Real data
fig = plt.figure(figsize=(30, 20))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Top row: Null Hypothesis
# Top-left: Forward-Only (Null)
ax1 = fig.add_subplot(gs[0, 0])
mask1 = forward_only_null_ordered == 0
sns.heatmap(forward_only_null_ordered, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask1, cbar_kws={'label': 'Fraction'}, ax=ax1,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
ax1.set_title('Forward-Only (Null Hypothesis)\n(A->B exists, B->A does not)', 
             fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Target Cell Type', fontsize=11)
ax1.set_ylabel('Source Cell Type', fontsize=11)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.tick_params(axis='y', rotation=0, labelsize=8)

# Top-middle: Reverse-Only (Null)
ax2 = fig.add_subplot(gs[0, 1])
mask2 = reverse_only_null_ordered == 0
sns.heatmap(reverse_only_null_ordered, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask2, cbar_kws={'label': 'Fraction'}, ax=ax2,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
ax2.set_title('Reverse-Only (Null Hypothesis)\n(B->A exists, A->B does not)', 
             fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Target Cell Type', fontsize=11)
ax2.set_ylabel('Source Cell Type', fontsize=11)
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.tick_params(axis='y', rotation=0, labelsize=8)

# Top-right: Symmetric (Null)
ax3 = fig.add_subplot(gs[0, 2])
mask3 = symmetric_null_ordered == 0
sns.heatmap(symmetric_null_ordered, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask3, cbar_kws={'label': 'Fraction'}, ax=ax3,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
ax3.set_title('Symmetric (Null Hypothesis)\n(Both A->B and B->A exist)', 
             fontsize=13, fontweight='bold', pad=15)
ax3.set_xlabel('Target Cell Type', fontsize=11)
ax3.set_ylabel('Source Cell Type', fontsize=11)
ax3.tick_params(axis='x', rotation=45, labelsize=8)
ax3.tick_params(axis='y', rotation=0, labelsize=8)

# Bottom row: Real Data
# Bottom-left: Forward-Only (Real)
ax4 = fig.add_subplot(gs[1, 0])
mask4 = forward_only_fraction_ordered == 0
sns.heatmap(forward_only_fraction_ordered, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask4, cbar_kws={'label': 'Fraction'}, ax=ax4,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
ax4.set_title('Forward-Only (Real Data)\n(A->B exists, B->A does not)', 
             fontsize=13, fontweight='bold', pad=15)
ax4.set_xlabel('Target Cell Type', fontsize=11)
ax4.set_ylabel('Source Cell Type', fontsize=11)
ax4.tick_params(axis='x', rotation=45, labelsize=8)
ax4.tick_params(axis='y', rotation=0, labelsize=8)

# Bottom-middle: Reverse-Only (Real)
ax5 = fig.add_subplot(gs[1, 1])
mask5 = reverse_only_fraction_ordered == 0
sns.heatmap(reverse_only_fraction_ordered, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask5, cbar_kws={'label': 'Fraction'}, ax=ax5,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
ax5.set_title('Reverse-Only (Real Data)\n(B->A exists, A->B does not)', 
             fontsize=13, fontweight='bold', pad=15)
ax5.set_xlabel('Target Cell Type', fontsize=11)
ax5.set_ylabel('Source Cell Type', fontsize=11)
ax5.tick_params(axis='x', rotation=45, labelsize=8)
ax5.tick_params(axis='y', rotation=0, labelsize=8)

# Bottom-right: Symmetric (Real)
ax6 = fig.add_subplot(gs[1, 2])
mask6 = symmetric_fraction_ordered == 0
sns.heatmap(symmetric_fraction_ordered, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask6, cbar_kws={'label': 'Fraction'}, ax=ax6,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
ax6.set_title('Symmetric (Real Data)\n(Both A->B and B->A exist)', 
             fontsize=13, fontweight='bold', pad=15)
ax6.set_xlabel('Target Cell Type', fontsize=11)
ax6.set_ylabel('Source Cell Type', fontsize=11)
ax6.tick_params(axis='x', rotation=45, labelsize=8)
ax6.tick_params(axis='y', rotation=0, labelsize=8)

plt.suptitle('Symmetric Connection Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.savefig(f'{figure_dir}/figure1_symmetric_connections_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figure_dir}/figure1_symmetric_connections_analysis.png")
plt.close()

# Summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Analysis shows fractions of connection types for each cell type pair.")
print(f"Each pair's fractions (forward-only, reverse-only, symmetric) sum to 1.0")

print(f"\nAll visualizations saved in '{figure_dir}' directory!")

