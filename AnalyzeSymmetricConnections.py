import pickle
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, spmatrix
from scipy.stats import binomtest
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
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

# Matrices for connection fractions (real data) - now 4 categories
forward_only_fraction_matrix = np.zeros((n_cell_types, n_cell_types))  # Fraction of forward-only connections
reverse_only_fraction_matrix = np.zeros((n_cell_types, n_cell_types))  # Fraction of reverse-only connections
symmetric_fraction_matrix = np.zeros((n_cell_types, n_cell_types))  # Fraction of symmetric connections
no_connection_fraction_matrix = np.zeros((n_cell_types, n_cell_types))  # Fraction with no connection

# Matrices for null hypothesis fractions
forward_only_null_matrix = np.zeros((n_cell_types, n_cell_types))
reverse_only_null_matrix = np.zeros((n_cell_types, n_cell_types))
symmetric_null_matrix = np.zeros((n_cell_types, n_cell_types))
no_connection_null_matrix = np.zeros((n_cell_types, n_cell_types))

# Store counts for statistical testing
forward_only_counts = np.zeros((n_cell_types, n_cell_types), dtype=int)
reverse_only_counts = np.zeros((n_cell_types, n_cell_types), dtype=int)
symmetric_counts = np.zeros((n_cell_types, n_cell_types), dtype=int)
no_connection_counts = np.zeros((n_cell_types, n_cell_types), dtype=int)
total_possible_pairs_counts = np.zeros((n_cell_types, n_cell_types), dtype=int)

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
        
        # Store counts for statistical testing
        forward_only_counts[i, j] = forward_only_pairs
        reverse_only_counts[i, j] = reverse_only_pairs
        symmetric_counts[i, j] = symmetric_pairs
        no_connection_counts[i, j] = no_connection_pairs
        total_possible_pairs_counts[i, j] = total_possible_pairs
        
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

forward_only_counts_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)), dtype=int)
reverse_only_counts_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)), dtype=int)
symmetric_counts_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)), dtype=int)
no_connection_counts_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)), dtype=int)
total_possible_pairs_counts_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)), dtype=int)

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
        forward_only_counts_ordered[i, j] = forward_only_counts[orig_i, orig_j]
        reverse_only_counts_ordered[i, j] = reverse_only_counts[orig_i, orig_j]
        symmetric_counts_ordered[i, j] = symmetric_counts[orig_i, orig_j]
        no_connection_counts_ordered[i, j] = no_connection_counts[orig_i, orig_j]
        total_possible_pairs_counts_ordered[i, j] = total_possible_pairs_counts[orig_i, orig_j]

# Print statistics
print("\n=== CONNECTION FRACTION STATISTICS ===")
print(f"{'Source':<20} {'Target':<20} {'Forward-Only':<15} {'Reverse-Only':<15} {'Symmetric':<15} {'No-Connection':<15} {'Sum':<15}")
print("-" * 120)

for i, source_type in enumerate(ordered_cell_types):
    for j, target_type in enumerate(ordered_cell_types):
        forward_only = forward_only_fraction_ordered[i, j]
        reverse_only = reverse_only_fraction_ordered[i, j]
        symmetric = symmetric_fraction_ordered[i, j]
        no_connection = no_connection_fraction_ordered[i, j]
        total = forward_only + reverse_only + symmetric + no_connection
        
        if total > 0:
            print(f"{source_type:<20} {target_type:<20} {forward_only:<15.4f} {reverse_only:<15.4f} {symmetric:<15.4f} {no_connection:<15.4f} {total:<15.4f}")

# Create visualizations
print("\n=== CREATING VISUALIZATIONS ===")
sns.set_style("whitegrid")

# Import LogNorm for log scale
from matplotlib.colors import LogNorm

# Single figure with all symmetry analyses: 4 rows x 4 columns
# Row 0: Connection diagrams
# Row 1: Null hypothesis (independent connections)
# Row 2: Real data
# Row 3: P-values
# Column order: No-Connection, Forward-Only, Reverse-Only, Symmetric
fig = plt.figure(figsize=(40, 32))
gs = fig.add_gridspec(4, 4, hspace=0.25, wspace=0.3, height_ratios=[0.15, 1, 1, 1])

# Helper function to draw connection diagram
def draw_connection_diagram(ax, connection_type):
    """Draw minimalist A-B connection diagram"""
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.3, 0.3)
    ax.axis('off')
    
    # Draw circles for A and B
    circle_a = Circle((0, 0), 0.15, fill=True, color='black', zorder=3)
    circle_b = Circle((1, 0), 0.15, fill=True, color='black', zorder=3)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    
    # Add labels
    ax.text(0, -0.25, 'A', ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(1, -0.25, 'B', ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Draw arrows based on connection type
    if connection_type == 'no_connection':
        # No arrows
        pass
    elif connection_type == 'forward_only':
        # A -> B
        arrow = FancyArrowPatch((0.15, 0), (0.85, 0), 
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black', zorder=2)
        ax.add_patch(arrow)
    elif connection_type == 'reverse_only':
        # B -> A
        arrow = FancyArrowPatch((0.85, 0), (0.15, 0), 
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black', zorder=2)
        ax.add_patch(arrow)
    elif connection_type == 'symmetric':
        # A <-> B (bidirectional)
        arrow1 = FancyArrowPatch((0.15, 0.1), (0.85, 0.1), 
                                arrowstyle='->', mutation_scale=20, 
                                linewidth=2, color='black', zorder=2)
        arrow2 = FancyArrowPatch((0.85, -0.1), (0.15, -0.1), 
                                arrowstyle='->', mutation_scale=20, 
                                linewidth=2, color='black', zorder=2)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)

# Row 0: Connection diagrams
ax_diag0 = fig.add_subplot(gs[0, 0])
draw_connection_diagram(ax_diag0, 'no_connection')

ax_diag1 = fig.add_subplot(gs[0, 1])
draw_connection_diagram(ax_diag1, 'forward_only')

ax_diag2 = fig.add_subplot(gs[0, 2])
draw_connection_diagram(ax_diag2, 'reverse_only')

ax_diag3 = fig.add_subplot(gs[0, 3])
draw_connection_diagram(ax_diag3, 'symmetric')

# Helper function to create log-scaled heatmap
def create_log_heatmap(data, mask, ax, title, xlabel, ylabel, vmin_offset=1e-6):
    # Apply log scale: log10(data + small_offset) to handle zeros for color mapping
    data_log = np.log10(data + vmin_offset)
    # Find min and max for colorbar (excluding masked values)
    valid_data = data[~mask]
    if len(valid_data) > 0 and valid_data.max() > 0:
        vmin_log = np.log10(valid_data[valid_data > 0].min() + vmin_offset)
        vmax_log = np.log10(valid_data.max() + vmin_offset)
    else:
        vmin_log = -6
        vmax_log = 0
    
    # Create annotation matrix with original (linear) values formatted as strings
    annot_data = np.empty(data.shape, dtype=object)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                annot_data[i, j] = ''
            else:
                annot_data[i, j] = f'{data[i, j]:.2f}'
    
    sns.heatmap(data_log, annot=annot_data, fmt='', cmap='YlOrRd', 
                xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
                mask=mask, cbar_kws={'label': 'log10(Fraction)'}, ax=ax,
                linewidths=0.5, linecolor='gray', vmin=vmin_log, vmax=vmax_log)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)

# Row 1: Null Hypothesis
# Column 0: No-Connection (Null)
ax1 = fig.add_subplot(gs[1, 0])
mask1 = no_connection_null_ordered == 0
create_log_heatmap(no_connection_null_ordered, mask1, ax1,
                   'No-Connection (Null)\n(Neither A->B nor B->A)',
                   'Target Cell Type', 'Source Cell Type')

# Column 1: Forward-Only (Null)
ax2 = fig.add_subplot(gs[1, 1])
mask2 = forward_only_null_ordered == 0
create_log_heatmap(forward_only_null_ordered, mask2, ax2,
                   'Forward-Only (Null)\n(A->B, not B->A)',
                   'Target Cell Type', 'Source Cell Type')

# Column 2: Reverse-Only (Null)
ax3 = fig.add_subplot(gs[1, 2])
mask3 = reverse_only_null_ordered == 0
create_log_heatmap(reverse_only_null_ordered, mask3, ax3,
                   'Reverse-Only (Null)\n(B->A, not A->B)',
                   'Target Cell Type', 'Source Cell Type')

# Column 3: Symmetric (Null)
ax4 = fig.add_subplot(gs[1, 3])
mask4 = symmetric_null_ordered == 0
create_log_heatmap(symmetric_null_ordered, mask4, ax4,
                   'Symmetric (Null)\n(Both A->B and B->A)',
                   'Target Cell Type', 'Source Cell Type')

# Row 2: Real Data
# Column 0: No-Connection (Real)
ax5 = fig.add_subplot(gs[2, 0])
mask5 = no_connection_fraction_ordered == 0
create_log_heatmap(no_connection_fraction_ordered, mask5, ax5,
                   'No-Connection (Real)\n(Neither A->B nor B->A)',
                   'Target Cell Type', 'Source Cell Type')

# Column 1: Forward-Only (Real)
ax6 = fig.add_subplot(gs[2, 1])
mask6 = forward_only_fraction_ordered == 0
create_log_heatmap(forward_only_fraction_ordered, mask6, ax6,
                   'Forward-Only (Real)\n(A->B, not B->A)',
                   'Target Cell Type', 'Source Cell Type')

# Column 2: Reverse-Only (Real)
ax7 = fig.add_subplot(gs[2, 2])
mask7 = reverse_only_fraction_ordered == 0
create_log_heatmap(reverse_only_fraction_ordered, mask7, ax7,
                   'Reverse-Only (Real)\n(B->A, not A->B)',
                   'Target Cell Type', 'Source Cell Type')

# Column 3: Symmetric (Real)
ax8 = fig.add_subplot(gs[2, 3])
mask8 = symmetric_fraction_ordered == 0
create_log_heatmap(symmetric_fraction_ordered, mask8, ax8,
                   'Symmetric (Real)\n(Both A->B and B->A)',
                   'Target Cell Type', 'Source Cell Type')

# Calculate p-values for statistical significance
print("\n=== CALCULATING P-VALUES ===")
forward_only_pvalue_matrix = np.ones((len(ordered_cell_types), len(ordered_cell_types)))
reverse_only_pvalue_matrix = np.ones((len(ordered_cell_types), len(ordered_cell_types)))
symmetric_pvalue_matrix = np.ones((len(ordered_cell_types), len(ordered_cell_types)))
no_connection_pvalue_matrix = np.ones((len(ordered_cell_types), len(ordered_cell_types)))

for i in range(len(ordered_cell_types)):
    for j in range(len(ordered_cell_types)):
        total = total_possible_pairs_counts_ordered[i, j]
        if total > 0:
            # Forward-only p-value
            observed_forward = forward_only_counts_ordered[i, j]
            expected_prob_forward = forward_only_null_ordered[i, j]
            if expected_prob_forward > 0 and expected_prob_forward < 1:
                result = binomtest(observed_forward, total, expected_prob_forward, alternative='two-sided')
                forward_only_pvalue_matrix[i, j] = result.pvalue
            elif expected_prob_forward == 0 and observed_forward > 0:
                forward_only_pvalue_matrix[i, j] = 0.0  # Significant deviation
            elif expected_prob_forward == 1 and observed_forward < total:
                forward_only_pvalue_matrix[i, j] = 0.0  # Significant deviation
            
            # Reverse-only p-value
            observed_reverse = reverse_only_counts_ordered[i, j]
            expected_prob_reverse = reverse_only_null_ordered[i, j]
            if expected_prob_reverse > 0 and expected_prob_reverse < 1:
                result = binomtest(observed_reverse, total, expected_prob_reverse, alternative='two-sided')
                reverse_only_pvalue_matrix[i, j] = result.pvalue
            elif expected_prob_reverse == 0 and observed_reverse > 0:
                reverse_only_pvalue_matrix[i, j] = 0.0
            elif expected_prob_reverse == 1 and observed_reverse < total:
                reverse_only_pvalue_matrix[i, j] = 0.0
            
            # Symmetric p-value
            observed_symmetric = symmetric_counts_ordered[i, j]
            expected_prob_symmetric = symmetric_null_ordered[i, j]
            if expected_prob_symmetric > 0 and expected_prob_symmetric < 1:
                result = binomtest(observed_symmetric, total, expected_prob_symmetric, alternative='two-sided')
                symmetric_pvalue_matrix[i, j] = result.pvalue
            elif expected_prob_symmetric == 0 and observed_symmetric > 0:
                symmetric_pvalue_matrix[i, j] = 0.0
            elif expected_prob_symmetric == 1 and observed_symmetric < total:
                symmetric_pvalue_matrix[i, j] = 0.0
            
            # No-connection p-value
            observed_no_conn = no_connection_counts_ordered[i, j]
            expected_prob_no_conn = no_connection_null_ordered[i, j]
            if expected_prob_no_conn > 0 and expected_prob_no_conn < 1:
                result = binomtest(observed_no_conn, total, expected_prob_no_conn, alternative='two-sided')
                no_connection_pvalue_matrix[i, j] = result.pvalue
            elif expected_prob_no_conn == 0 and observed_no_conn > 0:
                no_connection_pvalue_matrix[i, j] = 0.0
            elif expected_prob_no_conn == 1 and observed_no_conn < total:
                no_connection_pvalue_matrix[i, j] = 0.0

# Row 3: P-values (matching column order: No-Connection, Forward-Only, Reverse-Only, Symmetric)
# Column 0: No-Connection P-values
ax9 = fig.add_subplot(gs[3, 0])
mask9 = total_possible_pairs_counts_ordered == 0
pval_log_no_conn = np.where(no_connection_pvalue_matrix > 0, 
                            -np.log10(no_connection_pvalue_matrix + 1e-10),
                            np.full_like(no_connection_pvalue_matrix, 10))
exponent_no_conn = np.where(no_connection_pvalue_matrix > 0,
                           np.round(np.log10(no_connection_pvalue_matrix + 1e-10)).astype(int),
                           np.full_like(no_connection_pvalue_matrix, -10, dtype=int))
annot_no_conn = np.where(mask9, '', exponent_no_conn.astype(str))
sns.heatmap(pval_log_no_conn, annot=annot_no_conn, fmt='', cmap='RdYlGn_r', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask9, cbar_kws={'label': '-log10(p-value)'}, ax=ax9,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=5)
ax9.set_title('No-Connection P-values', fontsize=12, fontweight='bold', pad=10)
ax9.set_xlabel('Target Cell Type', fontsize=10)
ax9.set_ylabel('Source Cell Type', fontsize=10)
ax9.tick_params(axis='x', rotation=45, labelsize=7)
ax9.tick_params(axis='y', rotation=0, labelsize=7)

# Column 1: Forward-Only P-values
ax10 = fig.add_subplot(gs[3, 1])
mask10 = total_possible_pairs_counts_ordered == 0
pval_log_forward = np.where(forward_only_pvalue_matrix > 0, 
                            -np.log10(forward_only_pvalue_matrix + 1e-10),
                            np.full_like(forward_only_pvalue_matrix, 10))
exponent_forward = np.where(forward_only_pvalue_matrix > 0,
                           np.round(np.log10(forward_only_pvalue_matrix + 1e-10)).astype(int),
                           np.full_like(forward_only_pvalue_matrix, -10, dtype=int))
annot_forward = np.where(mask10, '', exponent_forward.astype(str))
sns.heatmap(pval_log_forward, annot=annot_forward, fmt='', cmap='RdYlGn_r', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask10, cbar_kws={'label': '-log10(p-value)'}, ax=ax10,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=5)
ax10.set_title('Forward-Only P-values', fontsize=12, fontweight='bold', pad=10)
ax10.set_xlabel('Target Cell Type', fontsize=10)
ax10.set_ylabel('Source Cell Type', fontsize=10)
ax10.tick_params(axis='x', rotation=45, labelsize=7)
ax10.tick_params(axis='y', rotation=0, labelsize=7)

# Column 2: Reverse-Only P-values
ax11 = fig.add_subplot(gs[3, 2])
mask11 = total_possible_pairs_counts_ordered == 0
pval_log_reverse = np.where(reverse_only_pvalue_matrix > 0, 
                            -np.log10(reverse_only_pvalue_matrix + 1e-10),
                            np.full_like(reverse_only_pvalue_matrix, 10))
exponent_reverse = np.where(reverse_only_pvalue_matrix > 0,
                           np.round(np.log10(reverse_only_pvalue_matrix + 1e-10)).astype(int),
                           np.full_like(reverse_only_pvalue_matrix, -10, dtype=int))
annot_reverse = np.where(mask11, '', exponent_reverse.astype(str))
sns.heatmap(pval_log_reverse, annot=annot_reverse, fmt='', cmap='RdYlGn_r', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask11, cbar_kws={'label': '-log10(p-value)'}, ax=ax11,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=5)
ax11.set_title('Reverse-Only P-values', fontsize=12, fontweight='bold', pad=10)
ax11.set_xlabel('Target Cell Type', fontsize=10)
ax11.set_ylabel('Source Cell Type', fontsize=10)
ax11.tick_params(axis='x', rotation=45, labelsize=7)
ax11.tick_params(axis='y', rotation=0, labelsize=7)

# Column 3: Symmetric P-values
ax12 = fig.add_subplot(gs[3, 3])
mask12 = total_possible_pairs_counts_ordered == 0
pval_log_symmetric = np.where(symmetric_pvalue_matrix > 0, 
                              -np.log10(symmetric_pvalue_matrix + 1e-10),
                              np.full_like(symmetric_pvalue_matrix, 10))
exponent_symmetric = np.where(symmetric_pvalue_matrix > 0,
                              np.round(np.log10(symmetric_pvalue_matrix + 1e-10)).astype(int),
                              np.full_like(symmetric_pvalue_matrix, -10, dtype=int))
annot_symmetric = np.where(mask12, '', exponent_symmetric.astype(str))
sns.heatmap(pval_log_symmetric, annot=annot_symmetric, fmt='', cmap='RdYlGn_r', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask12, cbar_kws={'label': '-log10(p-value)'}, ax=ax12,
            linewidths=0.5, linecolor='gray', vmin=0, vmax=5)
ax12.set_title('Symmetric P-values', fontsize=12, fontweight='bold', pad=10)
ax12.set_xlabel('Target Cell Type', fontsize=10)
ax12.set_ylabel('Source Cell Type', fontsize=10)
ax12.tick_params(axis='x', rotation=45, labelsize=7)
ax12.tick_params(axis='y', rotation=0, labelsize=7)

plt.suptitle('Symmetric Connection Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.savefig(f'{figure_dir}/figure1_symmetric_connections_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figure_dir}/figure1_symmetric_connections_analysis.png")
plt.close()


# Summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Analysis shows fractions of connection types for each cell type pair.")
print(f"Each pair's fractions (forward-only, reverse-only, symmetric) sum to 1.0")

print(f"\nAll visualizations saved in '{figure_dir}' directory!")

