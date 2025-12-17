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

# ============================================================================
# PART 1: ANALYZE DATA (or load if exists)
# ============================================================================
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
data_file = f'{data_dir}/pairwise_motifs_analysis.pkl'

if os.path.exists(data_file):
    print("=== LOADING EXISTING ANALYSIS DATA ===")
    with open(data_file, 'rb') as f:
        analysis_data = pickle.load(f)
    print(f"Loaded existing analysis data from: {data_file}")
else:
    print("=== RUNNING ANALYSIS (this may take a while) ===")
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

    print("\n=== ANALYZING PAIRWISE MOTIFS ===")

    # Initialize matrices
    n_cell_types = len(unique_cell_types)
    cell_type_to_idx = {ct: i for i, ct in enumerate(unique_cell_types)}

    # Matrices for connection fractions (real data) - 4 categories
    forward_only_fraction_matrix = np.zeros((n_cell_types, n_cell_types))
    reverse_only_fraction_matrix = np.zeros((n_cell_types, n_cell_types))
    symmetric_fraction_matrix = np.zeros((n_cell_types, n_cell_types))
    no_connection_fraction_matrix = np.zeros((n_cell_types, n_cell_types))

    # Matrices for null hypothesis fractions
    forward_only_null_matrix = np.zeros((n_cell_types, n_cell_types))
    reverse_only_null_matrix = np.zeros((n_cell_types, n_cell_types))
    symmetric_null_matrix = np.zeros((n_cell_types, n_cell_types))
    no_connection_null_matrix = np.zeros((n_cell_types, n_cell_types))

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
            no_connection_pairs = 0  # Neither A→B nor B→A exists
            
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
                    else:
                        no_connection_pairs += 1
            
            # Calculate total possible pairs
            total_possible_pairs = len(source_neurons) * len(target_neurons)
            
            # Calculate actual fractions (normalize to sum to 1 over all possible pairs)
            if total_possible_pairs > 0:
                forward_only_fraction_matrix[i, j] = forward_only_pairs / total_possible_pairs
                reverse_only_fraction_matrix[i, j] = reverse_only_pairs / total_possible_pairs
                symmetric_fraction_matrix[i, j] = symmetric_pairs / total_possible_pairs
                no_connection_fraction_matrix[i, j] = no_connection_pairs / total_possible_pairs
            else:
                forward_only_fraction_matrix[i, j] = 0
                reverse_only_fraction_matrix[i, j] = 0
                symmetric_fraction_matrix[i, j] = 0
                no_connection_fraction_matrix[i, j] = 0
            
            # Calculate connection probabilities for null hypothesis
            # p = probability of A->B, q = probability of B->A
            if total_possible_pairs > 0:
                p = (forward_only_pairs + symmetric_pairs) / total_possible_pairs
                q = (reverse_only_pairs + symmetric_pairs) / total_possible_pairs
            else:
                p = 0
                q = 0
            
            # Calculate null hypothesis fractions (assuming independent connections)
            # Under null: P(A->B only) = p*(1-q), P(B->A only) = (1-p)*q, P(both) = p*q, P(none) = (1-p)*(1-q)
            forward_only_null_matrix[i, j] = p * (1 - q)
            reverse_only_null_matrix[i, j] = (1 - p) * q
            symmetric_null_matrix[i, j] = p * q
            no_connection_null_matrix[i, j] = (1 - p) * (1 - q)

    # Reorder matrices to match ordered_cell_types
    ordered_to_original_idx = {ct: unique_cell_types.index(ct) for ct in ordered_cell_types}

    forward_only_fraction_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
    reverse_only_fraction_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
    symmetric_fraction_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
    no_connection_fraction_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))

    forward_only_null_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
    reverse_only_null_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
    symmetric_null_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
    no_connection_null_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))

    for i, ct_i in enumerate(ordered_cell_types):
        orig_i = ordered_to_original_idx[ct_i]
        for j, ct_j in enumerate(ordered_cell_types):
            orig_j = ordered_to_original_idx[ct_j]
            forward_only_fraction_ordered[i, j] = forward_only_fraction_matrix[orig_i, orig_j]
            reverse_only_fraction_ordered[i, j] = reverse_only_fraction_matrix[orig_i, orig_j]
            symmetric_fraction_ordered[i, j] = symmetric_fraction_matrix[orig_i, orig_j]
            no_connection_fraction_ordered[i, j] = no_connection_fraction_matrix[orig_i, orig_j]
            forward_only_null_ordered[i, j] = forward_only_null_matrix[orig_i, orig_j]
            reverse_only_null_ordered[i, j] = reverse_only_null_matrix[orig_i, orig_j]
            symmetric_null_ordered[i, j] = symmetric_null_matrix[orig_i, orig_j]
            no_connection_null_ordered[i, j] = no_connection_null_matrix[orig_i, orig_j]

    # Save analysis data
    print("\n=== SAVING ANALYSIS DATA ===")
    analysis_data = {
        'ordered_cell_types': ordered_cell_types,
        'forward_only_fraction_ordered': forward_only_fraction_ordered,
        'reverse_only_fraction_ordered': reverse_only_fraction_ordered,
        'symmetric_fraction_ordered': symmetric_fraction_ordered,
        'no_connection_fraction_ordered': no_connection_fraction_ordered,
        'forward_only_null_ordered': forward_only_null_ordered,
        'reverse_only_null_ordered': reverse_only_null_ordered,
        'symmetric_null_ordered': symmetric_null_ordered,
        'no_connection_null_ordered': no_connection_null_ordered,
    }

    with open(data_file, 'wb') as f:
        pickle.dump(analysis_data, f)
    print(f"Saved analysis data to: {data_file}")

# ============================================================================
# PART 2: CREATE FIGURES
# ============================================================================
print("\n=== LOADING DATA FOR VISUALIZATION ===")
with open(data_file, 'rb') as f:
    analysis_data = pickle.load(f)

ordered_cell_types = analysis_data['ordered_cell_types']
forward_only_fraction_ordered = analysis_data['forward_only_fraction_ordered']
reverse_only_fraction_ordered = analysis_data['reverse_only_fraction_ordered']
symmetric_fraction_ordered = analysis_data['symmetric_fraction_ordered']
no_connection_fraction_ordered = analysis_data['no_connection_fraction_ordered']
forward_only_null_ordered = analysis_data['forward_only_null_ordered']
reverse_only_null_ordered = analysis_data['reverse_only_null_ordered']
symmetric_null_ordered = analysis_data['symmetric_null_ordered']
no_connection_null_ordered = analysis_data['no_connection_null_ordered']

print(f"Loaded data for {len(ordered_cell_types)} cell types")

print("\n=== CREATING GROUPED BAR CHARTS ===")
# Create figure directory
figure_dir = 'figures/both_known'
os.makedirs(figure_dir, exist_ok=True)

sns.set_style("whitegrid")

# Calculate number of pairs to plot
n_pairs = len(ordered_cell_types) * len(ordered_cell_types)
n_cols = int(np.ceil(np.sqrt(n_pairs)))  # Square root for roughly square layout
n_rows = int(np.ceil(n_pairs / n_cols))

# Create figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
axes = axes.flatten() if n_pairs > 1 else [axes] if n_rows == 1 else axes.flatten()

# Motif labels
motif_labels = ['No Connection', 'Forward-Only\n(A→B)', 'Reverse-Only\n(B→A)', 'Symmetric\n(A↔B)']
x = np.arange(len(motif_labels))
width = 0.35  # Width of bars

pair_idx = 0
for i, source_type in enumerate(ordered_cell_types):
    for j, target_type in enumerate(ordered_cell_types):
        ax = axes[pair_idx]
        
        # Get expected (real) values
        expected_values = [
            no_connection_fraction_ordered[i, j],
            forward_only_fraction_ordered[i, j],
            reverse_only_fraction_ordered[i, j],
            symmetric_fraction_ordered[i, j]
        ]
        
        # Get null hypothesis values
        null_values = [
            no_connection_null_ordered[i, j],
            forward_only_null_ordered[i, j],
            reverse_only_null_ordered[i, j],
            symmetric_null_ordered[i, j]
        ]
        
        # Add small offset to avoid zero values for log scale
        epsilon = 1e-6
        expected_values_plot = [max(v, epsilon) for v in expected_values]
        null_values_plot = [max(v, epsilon) for v in null_values]
        
        # Create grouped bars
        bars1 = ax.bar(x - width/2, expected_values_plot, width, label='Expected', 
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, null_values_plot, width, label='Null Hypothesis', 
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Set log scale BEFORE adding labels (so labels are positioned correctly)
        ax.set_yscale('log')
        
        # Add value labels on bars (using original linear values for annotations)
        for bars, orig_values in [(bars1, expected_values), (bars2, null_values)]:
            for bar, orig_val in zip(bars, orig_values):
                if orig_val > 0.001:  # Only label if significant
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{orig_val:.3f}',
                           ha='center', va='bottom', fontsize=7)
        
        # Customize plot
        ax.set_xlabel('Motif Type', fontsize=9)
        ax.set_ylabel('Fraction (log scale)', fontsize=9)
        ax.set_title(f'{source_type} → {target_type}', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(motif_labels, fontsize=8, rotation=45, ha='right')
        # Set reasonable y-limits for log scale
        all_values = [v for v in expected_values + null_values if v > 0]
        if all_values:
            y_min = min(all_values) * 0.5
            y_max = max(all_values) * 2
            ax.set_ylim([y_min, y_max])
        else:
            ax.set_ylim([1e-6, 1.0])
        ax.grid(axis='y', alpha=0.3, which='both')
        ax.legend(fontsize=8, loc='upper right')
        
        pair_idx += 1

# Hide unused subplots
for idx in range(pair_idx, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Pairwise Motif Analysis: Expected vs. Null Hypothesis\n(Both Known: Synapses Between Neurons in Dataset)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(f'{figure_dir}/figure5_pairwise_motifs.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figure_dir}/figure5_pairwise_motifs.png")
plt.close()

print(f"\nAll visualizations saved in '{figure_dir}' directory!")

