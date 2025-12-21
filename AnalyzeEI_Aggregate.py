"""
Analyze connectome data aggregated by E/I neurons for BOTH_KNOWN case.

This script creates comprehensive figures showing:
- Full connectivity matrices (E->E, E->I, I->E, I->I)
- Symmetry analysis (forward-only, reverse-only, symmetric, no-connection)
- All relevant analyses from both_known case aggregated by E/I
"""

import pickle
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, spmatrix
from scipy.stats import binomtest
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import FancyArrowPatch, Circle

def load_connectivity(path: str, name: str) -> tuple[spmatrix, dict]:
    sparse_matrix = load_npz(f'{path}/{name}_matrix.npz')
    with open(f'{path}/{name}_mapping.pkl', 'rb') as fp:
        mapping = pickle.load(fp)
    return sparse_matrix, mapping

print("=== ANALYZING E/I AGGREGATE (BOTH KNOWN) ===\n")

# Load data
CONNECTIVITY_DIR = '.'
syn_mat_sparse, mapping = load_connectivity(CONNECTIVITY_DIR, 'network_synapses')
neurons_df = pd.read_csv('connectome_neurons.csv')

print(f"Loaded {len(neurons_df)} neurons")

# Create mappings
root_id_to_clf_type = dict(zip(neurons_df['root_id'], neurons_df['clf_type']))
matrix_idx_to_clf_type = {}
for idx, root_id in mapping.items():
    matrix_idx_to_clf_type[idx] = root_id_to_clf_type.get(root_id, 'Unknown')

# Get E/I counts
clf_type_counts = Counter(matrix_idx_to_clf_type.values())
e_count = clf_type_counts.get('E', 0)
i_count = clf_type_counts.get('I', 0)
ei_labels = ['E', 'I']

print(f"E neurons: {e_count}")
print(f"I neurons: {i_count}")

# Create figure directory
figure_dir = 'figures/ei_aggregate'
os.makedirs(figure_dir, exist_ok=True)

# ============================================================================
# COMPUTE CONNECTIVITY MATRICES (E->E, E->I, I->E, I->I)
# ============================================================================
print("\n=== COMPUTING CONNECTIVITY MATRICES ===")

ei_matrix = np.zeros((2, 2), dtype=int)  # [E, I] x [E, I]
ei_synapse_matrix = np.zeros((2, 2))

# Get E and I neuron indices
e_neurons = [idx for idx, clf_type in matrix_idx_to_clf_type.items() if clf_type == 'E']
i_neurons = [idx for idx, clf_type in matrix_idx_to_clf_type.items() if clf_type == 'I']

# Aggregate connections
for i in range(syn_mat_sparse.shape[0]):
    source_clf_type = matrix_idx_to_clf_type.get(i, 'Unknown')
    if source_clf_type not in ['E', 'I']:
        continue
    
    source_idx = 0 if source_clf_type == 'E' else 1
    row = syn_mat_sparse[i, :]
    if row.nnz > 0:
        targets = row.indices
        weights = row.data
        for target_idx, weight in zip(targets, weights):
            target_clf_type = matrix_idx_to_clf_type.get(target_idx, 'Unknown')
            if target_clf_type not in ['E', 'I']:
                continue
            target_type_idx = 0 if target_clf_type == 'E' else 1
            ei_matrix[source_idx, target_type_idx] += 1
            ei_synapse_matrix[source_idx, target_type_idx] += weight

# Compute probability and mean synapses matrices
prob_matrix = np.zeros((2, 2))
mean_syn_matrix = np.zeros((2, 2))

for i, source_type in enumerate(ei_labels):
    for j, target_type in enumerate(ei_labels):
        n_connections = ei_matrix[i, j]
        source_count = e_count if i == 0 else i_count
        target_count = e_count if j == 0 else i_count
        
        if source_count > 0 and target_count > 0:
            if i == j:
                possible_connections = source_count * (target_count - 1)
            else:
                possible_connections = source_count * target_count
            if possible_connections > 0:
                prob_matrix[i, j] = n_connections / possible_connections
        
        n_conn = ei_matrix[i, j]
        n_syn = ei_synapse_matrix[i, j]
        if n_conn > 0:
            mean_syn_matrix[i, j] = n_syn / n_conn

# ============================================================================
# COMPUTE SYMMETRY ANALYSIS
# ============================================================================
print("\n=== COMPUTING SYMMETRY ANALYSIS ===")

# Matrices for connection fractions
forward_only_fraction = np.zeros((2, 2))
reverse_only_fraction = np.zeros((2, 2))
symmetric_fraction = np.zeros((2, 2))
no_connection_fraction = np.zeros((2, 2))

# Matrices for null hypothesis
forward_only_null = np.zeros((2, 2))
reverse_only_null = np.zeros((2, 2))
symmetric_null = np.zeros((2, 2))
no_connection_null = np.zeros((2, 2))

# Matrices for counts
forward_only_counts = np.zeros((2, 2), dtype=int)
reverse_only_counts = np.zeros((2, 2), dtype=int)
symmetric_counts = np.zeros((2, 2), dtype=int)
no_connection_counts = np.zeros((2, 2), dtype=int)
total_pairs_counts = np.zeros((2, 2), dtype=int)

for i, source_type in enumerate(ei_labels):
    source_neurons = e_neurons if i == 0 else i_neurons
    for j, target_type in enumerate(ei_labels):
        target_neurons = e_neurons if j == 0 else i_neurons
        
        forward_only_pairs = 0
        reverse_only_pairs = 0
        symmetric_pairs = 0
        no_connection_pairs = 0
        
        for source_idx in source_neurons:
            for target_idx in target_neurons:
                source_row = syn_mat_sparse[source_idx, :]
                has_forward = False
                if source_row.nnz > 0 and target_idx in source_row.indices:
                    has_forward = True
                
                target_row = syn_mat_sparse[target_idx, :]
                has_reverse = False
                if target_row.nnz > 0 and source_idx in target_row.indices:
                    has_reverse = True
                
                if has_forward and has_reverse:
                    symmetric_pairs += 1
                elif has_forward:
                    forward_only_pairs += 1
                elif has_reverse:
                    reverse_only_pairs += 1
                else:
                    no_connection_pairs += 1
        
        total_possible_pairs = len(source_neurons) * len(target_neurons)
        
        forward_only_counts[i, j] = forward_only_pairs
        reverse_only_counts[i, j] = reverse_only_pairs
        symmetric_counts[i, j] = symmetric_pairs
        no_connection_counts[i, j] = no_connection_pairs
        total_pairs_counts[i, j] = total_possible_pairs
        
        if total_possible_pairs > 0:
            forward_only_fraction[i, j] = forward_only_pairs / total_possible_pairs
            reverse_only_fraction[i, j] = reverse_only_pairs / total_possible_pairs
            symmetric_fraction[i, j] = symmetric_pairs / total_possible_pairs
            no_connection_fraction[i, j] = no_connection_pairs / total_possible_pairs
        
        # Null hypothesis
        if total_possible_pairs > 0:
            p = (forward_only_pairs + symmetric_pairs) / total_possible_pairs
            q = (reverse_only_pairs + symmetric_pairs) / total_possible_pairs
        else:
            p = 0
            q = 0
        
        forward_only_null[i, j] = p * (1 - q)
        reverse_only_null[i, j] = (1 - p) * q
        symmetric_null[i, j] = p * q
        no_connection_null[i, j] = (1 - p) * (1 - q)

# ============================================================================
# CREATE CONNECTIVITY MATRICES FIGURE
# ============================================================================
print("\n=== CREATING CONNECTIVITY MATRICES FIGURE ===")
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Number of Connections
conn_matrix_log = np.log10(ei_matrix + 1)
mask1 = ei_matrix == 0
annot_matrix = ei_matrix.astype(int)
sns.heatmap(conn_matrix_log, annot=annot_matrix, fmt='d', cmap='YlOrRd',
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask1, cbar_kws={'label': 'Log10(Connections + 1)'}, ax=axes[0, 0],
            linewidths=1, linecolor='black', square=True)
axes[0, 0].set_title('Number of Connections (log scale)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Target', fontsize=11)
axes[0, 0].set_ylabel('Source', fontsize=11)

# Connection Probability
mask2 = prob_matrix == 0
sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='viridis',
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask2, cbar_kws={'label': 'Connection Probability'}, ax=axes[0, 1],
            linewidths=1, linecolor='black', square=True)
axes[0, 1].set_title('Connection Probability', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Target', fontsize=11)
axes[0, 1].set_ylabel('Source', fontsize=11)

# Mean Synapses per Connection
mask3 = mean_syn_matrix == 0
sns.heatmap(mean_syn_matrix, annot=True, fmt='.2f', cmap='plasma',
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask3, cbar_kws={'label': 'Mean Synapses per Connection'}, ax=axes[1, 0],
            linewidths=1, linecolor='black', square=True)
axes[1, 0].set_title('Mean Synapses per Connection', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Target', fontsize=11)
axes[1, 0].set_ylabel('Source', fontsize=11)

# Total Synapses
syn_matrix_log = np.log10(ei_synapse_matrix + 1)
mask4 = ei_synapse_matrix == 0
annot_syn = ei_synapse_matrix.astype(int)
sns.heatmap(syn_matrix_log, annot=annot_syn, fmt='d', cmap='coolwarm',
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask4, cbar_kws={'label': 'Log10(Total Synapses + 1)'}, ax=axes[1, 1],
            linewidths=1, linecolor='black', square=True)
axes[1, 1].set_title('Total Synapses (log scale)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Target', fontsize=11)
axes[1, 1].set_ylabel('Source', fontsize=11)

plt.suptitle('E/I Connectivity Matrices (Both Known)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f'{figure_dir}/figure1_connectivity_matrices.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figure_dir}/figure1_connectivity_matrices.png")
plt.close()

# ============================================================================
# CREATE SYMMETRY ANALYSIS FIGURE
# ============================================================================
print("\n=== CREATING SYMMETRY ANALYSIS FIGURE ===")

# Calculate p-values
forward_only_pvalue = np.ones((2, 2))
reverse_only_pvalue = np.ones((2, 2))
symmetric_pvalue = np.ones((2, 2))
no_connection_pvalue = np.ones((2, 2))

for i in range(2):
    for j in range(2):
        total = total_pairs_counts[i, j]
        if total > 0:
            # Forward-only
            observed = forward_only_counts[i, j]
            expected_prob = forward_only_null[i, j]
            if expected_prob > 0 and expected_prob < 1:
                result = binomtest(observed, total, expected_prob, alternative='two-sided')
                forward_only_pvalue[i, j] = result.pvalue
            
            # Reverse-only
            observed = reverse_only_counts[i, j]
            expected_prob = reverse_only_null[i, j]
            if expected_prob > 0 and expected_prob < 1:
                result = binomtest(observed, total, expected_prob, alternative='two-sided')
                reverse_only_pvalue[i, j] = result.pvalue
            
            # Symmetric
            observed = symmetric_counts[i, j]
            expected_prob = symmetric_null[i, j]
            if expected_prob > 0 and expected_prob < 1:
                result = binomtest(observed, total, expected_prob, alternative='two-sided')
                symmetric_pvalue[i, j] = result.pvalue
            
            # No-connection
            observed = no_connection_counts[i, j]
            expected_prob = no_connection_null[i, j]
            if expected_prob > 0 and expected_prob < 1:
                result = binomtest(observed, total, expected_prob, alternative='two-sided')
                no_connection_pvalue[i, j] = result.pvalue

# Create figure: 5 rows x 4 columns (added ratio row)
fig = plt.figure(figsize=(20, 20))
gs = fig.add_gridspec(5, 4, hspace=0.25, wspace=0.3, height_ratios=[0.15, 1, 1, 1, 1])

# Helper function to draw connection diagram
def draw_connection_diagram(ax, connection_type):
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.3, 0.3)
    ax.axis('off')
    
    circle_a = Circle((0, 0), 0.15, fill=True, color='black', zorder=3)
    circle_b = Circle((1, 0), 0.15, fill=True, color='black', zorder=3)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    
    ax.text(0, -0.25, 'A', ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(1, -0.25, 'B', ha='center', va='top', fontsize=14, fontweight='bold')
    
    if connection_type == 'no_connection':
        pass
    elif connection_type == 'forward_only':
        arrow = FancyArrowPatch((0.15, 0), (0.85, 0),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black', zorder=2)
        ax.add_patch(arrow)
    elif connection_type == 'reverse_only':
        arrow = FancyArrowPatch((0.85, 0), (0.15, 0),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black', zorder=2)
        ax.add_patch(arrow)
    elif connection_type == 'symmetric':
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

# Helper function to create log-scaled heatmap with raw numbers
def create_log_heatmap(data, mask, ax, title, counts, vmin_log, vmax_log, vmin_offset=1e-6):
    data_log = np.log10(data + vmin_offset)
    
    annot_data = np.empty(data.shape, dtype=object)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                annot_data[i, j] = ''
            else:
                annot_data[i, j] = f'{data[i, j]:.3f}\n({int(counts[i, j])})'
    
    sns.heatmap(data_log, annot=annot_data, fmt='', cmap='YlOrRd',
                xticklabels=ei_labels, yticklabels=ei_labels,
                mask=mask, cbar_kws={'label': 'log10(Fraction)'}, ax=ax,
                linewidths=1, linecolor='black', square=True, vmin=vmin_log, vmax=vmax_log)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Target', fontsize=10)
    ax.set_ylabel('Source', fontsize=10)

# Calculate expected counts for null hypothesis
no_conn_null_counts = (no_connection_null * total_pairs_counts).astype(int)
forward_null_counts = (forward_only_null * total_pairs_counts).astype(int)
reverse_null_counts = (reverse_only_null * total_pairs_counts).astype(int)
symmetric_null_counts = (symmetric_null * total_pairs_counts).astype(int)

# Calculate global color scale ranges for each ROW (for consistent scaling within rows)
# Row 1 & 2: All fraction data combined
all_fractions = np.concatenate([
    no_connection_null.flatten(), no_connection_fraction.flatten(),
    forward_only_null.flatten(), forward_only_fraction.flatten(),
    reverse_only_null.flatten(), reverse_only_fraction.flatten(),
    symmetric_null.flatten(), symmetric_fraction.flatten()
])
all_fractions = all_fractions[all_fractions > 0]
fraction_vmin = np.log10(all_fractions.min() + 1e-6) if len(all_fractions) > 0 else -6
fraction_vmax = np.log10(all_fractions.max() + 1e-6) if len(all_fractions) > 0 else 0

# Row 1: Null Hypothesis (with expected counts)
ax1 = fig.add_subplot(gs[1, 0])
mask1 = no_connection_null == 0
create_log_heatmap(no_connection_null, mask1, ax1, 'No-Connection (Null)',
                  counts=no_conn_null_counts, vmin_log=fraction_vmin, vmax_log=fraction_vmax)
ax2 = fig.add_subplot(gs[1, 1])
mask2 = forward_only_null == 0
create_log_heatmap(forward_only_null, mask2, ax2, 'Forward-Only (Null)',
                  counts=forward_null_counts, vmin_log=fraction_vmin, vmax_log=fraction_vmax)
ax3 = fig.add_subplot(gs[1, 2])
mask3 = reverse_only_null == 0
create_log_heatmap(reverse_only_null, mask3, ax3, 'Reverse-Only (Null)',
                  counts=reverse_null_counts, vmin_log=fraction_vmin, vmax_log=fraction_vmax)
ax4 = fig.add_subplot(gs[1, 3])
mask4 = symmetric_null == 0
create_log_heatmap(symmetric_null, mask4, ax4, 'Symmetric (Null)',
                  counts=symmetric_null_counts, vmin_log=fraction_vmin, vmax_log=fraction_vmax)

# Row 2: Real Data (with observed counts)
ax5 = fig.add_subplot(gs[2, 0])
mask5 = no_connection_fraction == 0
create_log_heatmap(no_connection_fraction, mask5, ax5, 'No-Connection (Real)',
                  counts=no_connection_counts, vmin_log=fraction_vmin, vmax_log=fraction_vmax)
ax6 = fig.add_subplot(gs[2, 1])
mask6 = forward_only_fraction == 0
create_log_heatmap(forward_only_fraction, mask6, ax6, 'Forward-Only (Real)',
                  counts=forward_only_counts, vmin_log=fraction_vmin, vmax_log=fraction_vmax)
ax7 = fig.add_subplot(gs[2, 2])
mask7 = reverse_only_fraction == 0
create_log_heatmap(reverse_only_fraction, mask7, ax7, 'Reverse-Only (Real)',
                  counts=reverse_only_counts, vmin_log=fraction_vmin, vmax_log=fraction_vmax)
ax8 = fig.add_subplot(gs[2, 3])
mask8 = symmetric_fraction == 0
create_log_heatmap(symmetric_fraction, mask8, ax8, 'Symmetric (Real)',
                  counts=symmetric_counts, vmin_log=fraction_vmin, vmax_log=fraction_vmax)

# Row 3: Ratio (Real/Null) - LINEAR scale, not log
# Calculate ratios
no_conn_ratio = np.divide(no_connection_fraction, no_connection_null, 
                          out=np.zeros_like(no_connection_fraction), 
                          where=no_connection_null != 0)
forward_ratio = np.divide(forward_only_fraction, forward_only_null,
                         out=np.zeros_like(forward_only_fraction),
                         where=forward_only_null != 0)
reverse_ratio = np.divide(reverse_only_fraction, reverse_only_null,
                         out=np.zeros_like(reverse_only_fraction),
                         where=reverse_only_null != 0)
symmetric_ratio = np.divide(symmetric_fraction, symmetric_null,
                           out=np.zeros_like(symmetric_fraction),
                           where=symmetric_null != 0)

# Calculate expected counts for ratio annotation (reuse from earlier)
no_conn_expected_counts = no_conn_null_counts.astype(float)
forward_expected_counts = forward_null_counts.astype(float)
reverse_expected_counts = reverse_null_counts.astype(float)
symmetric_expected_counts = symmetric_null_counts.astype(float)

# Calculate global ratio scale for entire row
all_ratios = np.concatenate([no_conn_ratio.flatten(), forward_ratio.flatten(),
                             reverse_ratio.flatten(), symmetric_ratio.flatten()])
all_ratios = all_ratios[all_ratios > 0]
ratio_vmin = 0
ratio_vmax = all_ratios.max() if len(all_ratios) > 0 else 2

ax_ratio1 = fig.add_subplot(gs[3, 0])
mask_ratio1 = (no_connection_null == 0) | (no_connection_fraction == 0)
annot_ratio1 = np.empty(no_conn_ratio.shape, dtype=object)
for i in range(no_conn_ratio.shape[0]):
    for j in range(no_conn_ratio.shape[1]):
        if mask_ratio1[i, j]:
            annot_ratio1[i, j] = ''
        else:
            annot_ratio1[i, j] = f'{no_conn_ratio[i, j]:.2f}\n({int(no_connection_counts[i, j])}/{int(no_conn_expected_counts[i, j])})'
sns.heatmap(no_conn_ratio, annot=annot_ratio1, fmt='', cmap='RdBu_r', center=1,
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask_ratio1, cbar_kws={'label': 'Ratio (Real/Null)'}, ax=ax_ratio1,
            linewidths=1, linecolor='black', square=True, vmin=ratio_vmin, vmax=ratio_vmax)
ax_ratio1.set_title('No-Connection Ratio\n(Real/Null)', fontsize=12, fontweight='bold', pad=10)
ax_ratio1.set_xlabel('Target', fontsize=10)
ax_ratio1.set_ylabel('Source', fontsize=10)

ax_ratio2 = fig.add_subplot(gs[3, 1])
mask_ratio2 = (forward_only_null == 0) | (forward_only_fraction == 0)
annot_ratio2 = np.empty(forward_ratio.shape, dtype=object)
for i in range(forward_ratio.shape[0]):
    for j in range(forward_ratio.shape[1]):
        if mask_ratio2[i, j]:
            annot_ratio2[i, j] = ''
        else:
            annot_ratio2[i, j] = f'{forward_ratio[i, j]:.2f}\n({int(forward_only_counts[i, j])}/{int(forward_expected_counts[i, j])})'
sns.heatmap(forward_ratio, annot=annot_ratio2, fmt='', cmap='RdBu_r', center=1,
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask_ratio2, cbar_kws={'label': 'Ratio (Real/Null)'}, ax=ax_ratio2,
            linewidths=1, linecolor='black', square=True, vmin=ratio_vmin, vmax=ratio_vmax)
ax_ratio2.set_title('Forward-Only Ratio\n(Real/Null)', fontsize=12, fontweight='bold', pad=10)
ax_ratio2.set_xlabel('Target', fontsize=10)
ax_ratio2.set_ylabel('Source', fontsize=10)

ax_ratio3 = fig.add_subplot(gs[3, 2])
mask_ratio3 = (reverse_only_null == 0) | (reverse_only_fraction == 0)
annot_ratio3 = np.empty(reverse_ratio.shape, dtype=object)
for i in range(reverse_ratio.shape[0]):
    for j in range(reverse_ratio.shape[1]):
        if mask_ratio3[i, j]:
            annot_ratio3[i, j] = ''
        else:
            annot_ratio3[i, j] = f'{reverse_ratio[i, j]:.2f}\n({int(reverse_only_counts[i, j])}/{int(reverse_expected_counts[i, j])})'
sns.heatmap(reverse_ratio, annot=annot_ratio3, fmt='', cmap='RdBu_r', center=1,
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask_ratio3, cbar_kws={'label': 'Ratio (Real/Null)'}, ax=ax_ratio3,
            linewidths=1, linecolor='black', square=True, vmin=ratio_vmin, vmax=ratio_vmax)
ax_ratio3.set_title('Reverse-Only Ratio\n(Real/Null)', fontsize=12, fontweight='bold', pad=10)
ax_ratio3.set_xlabel('Target', fontsize=10)
ax_ratio3.set_ylabel('Source', fontsize=10)

ax_ratio4 = fig.add_subplot(gs[3, 3])
mask_ratio4 = (symmetric_null == 0) | (symmetric_fraction == 0)
annot_ratio4 = np.empty(symmetric_ratio.shape, dtype=object)
for i in range(symmetric_ratio.shape[0]):
    for j in range(symmetric_ratio.shape[1]):
        if mask_ratio4[i, j]:
            annot_ratio4[i, j] = ''
        else:
            annot_ratio4[i, j] = f'{symmetric_ratio[i, j]:.2f}\n({int(symmetric_counts[i, j])}/{int(symmetric_expected_counts[i, j])})'
sns.heatmap(symmetric_ratio, annot=annot_ratio4, fmt='', cmap='RdBu_r', center=1,
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask_ratio4, cbar_kws={'label': 'Ratio (Real/Null)'}, ax=ax_ratio4,
            linewidths=1, linecolor='black', square=True, vmin=ratio_vmin, vmax=ratio_vmax)
ax_ratio4.set_title('Symmetric Ratio\n(Real/Null)', fontsize=12, fontweight='bold', pad=10)
ax_ratio4.set_xlabel('Target', fontsize=10)
ax_ratio4.set_ylabel('Source', fontsize=10)

# Row 4: P-values - calculate all -log10 values first for proper scaling
# Replace 0 p-values with a very small number for log calculation
pval_epsilon = 1e-300

no_conn_pval_log = -np.log10(np.maximum(no_connection_pvalue, pval_epsilon))
forward_pval_log = -np.log10(np.maximum(forward_only_pvalue, pval_epsilon))
reverse_pval_log = -np.log10(np.maximum(reverse_only_pvalue, pval_epsilon))
symmetric_pval_log = -np.log10(np.maximum(symmetric_pvalue, pval_epsilon))

# Find actual max for proper scaling across ALL p-value heatmaps in this row
all_pval_logs = np.concatenate([no_conn_pval_log.flatten(), forward_pval_log.flatten(),
                                reverse_pval_log.flatten(), symmetric_pval_log.flatten()])
max_pval_log = all_pval_logs.max()
print(f"P-value range: max -log10={max_pval_log:.1f}")

mask9 = total_pairs_counts == 0

ax9 = fig.add_subplot(gs[4, 0])
exponent_no_conn = np.round(np.log10(np.maximum(no_connection_pvalue, pval_epsilon))).astype(int)
annot_no_conn = np.where(mask9, '', exponent_no_conn.astype(str))
sns.heatmap(no_conn_pval_log, annot=annot_no_conn, fmt='', cmap='RdYlGn_r',
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask9, cbar_kws={'label': '-log10(p-value)'}, ax=ax9,
            linewidths=1, linecolor='black', square=True, vmin=0, vmax=max_pval_log)
ax9.set_title('No-Connection P-values', fontsize=12, fontweight='bold', pad=10)
ax9.set_xlabel('Target', fontsize=10)
ax9.set_ylabel('Source', fontsize=10)

ax10 = fig.add_subplot(gs[4, 1])
exponent_forward = np.round(np.log10(np.maximum(forward_only_pvalue, pval_epsilon))).astype(int)
annot_forward = np.where(mask9, '', exponent_forward.astype(str))
sns.heatmap(forward_pval_log, annot=annot_forward, fmt='', cmap='RdYlGn_r',
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask9, cbar_kws={'label': '-log10(p-value)'}, ax=ax10,
            linewidths=1, linecolor='black', square=True, vmin=0, vmax=max_pval_log)
ax10.set_title('Forward-Only P-values', fontsize=12, fontweight='bold', pad=10)
ax10.set_xlabel('Target', fontsize=10)
ax10.set_ylabel('Source', fontsize=10)

ax11 = fig.add_subplot(gs[4, 2])
exponent_reverse = np.round(np.log10(np.maximum(reverse_only_pvalue, pval_epsilon))).astype(int)
annot_reverse = np.where(mask9, '', exponent_reverse.astype(str))
sns.heatmap(reverse_pval_log, annot=annot_reverse, fmt='', cmap='RdYlGn_r',
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask9, cbar_kws={'label': '-log10(p-value)'}, ax=ax11,
            linewidths=1, linecolor='black', square=True, vmin=0, vmax=max_pval_log)
ax11.set_title('Reverse-Only P-values', fontsize=12, fontweight='bold', pad=10)
ax11.set_xlabel('Target', fontsize=10)
ax11.set_ylabel('Source', fontsize=10)

ax12 = fig.add_subplot(gs[4, 3])
exponent_symmetric = np.round(np.log10(np.maximum(symmetric_pvalue, pval_epsilon))).astype(int)
annot_symmetric = np.where(mask9, '', exponent_symmetric.astype(str))
sns.heatmap(symmetric_pval_log, annot=annot_symmetric, fmt='', cmap='RdYlGn_r',
            xticklabels=ei_labels, yticklabels=ei_labels,
            mask=mask9, cbar_kws={'label': '-log10(p-value)'}, ax=ax12,
            linewidths=1, linecolor='black', square=True, vmin=0, vmax=max_pval_log)
ax12.set_title('Symmetric P-values', fontsize=12, fontweight='bold', pad=10)
ax12.set_xlabel('Target', fontsize=10)
ax12.set_ylabel('Source', fontsize=10)

plt.suptitle('E/I Symmetric Connection Analysis (Both Known)', fontsize=18, fontweight='bold', y=0.995)
plt.savefig(f'{figure_dir}/figure2_symmetric_connections.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figure_dir}/figure2_symmetric_connections.png")
plt.close()

# ============================================================================
# CREATE BAR PLOTS FOR SYMMETRY ANALYSIS
# ============================================================================
print("\n=== CREATING SYMMETRY BAR PLOTS ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

motif_labels = ['No Connection', 'Forward-Only\n(A→B)', 'Reverse-Only\n(B→A)', 'Symmetric\n(A↔B)']
x = np.arange(len(motif_labels))
width = 0.35

pair_idx = 0
for i, source_type in enumerate(ei_labels):
    for j, target_type in enumerate(ei_labels):
        ax = axes[pair_idx]
        
        # Get data (observed) values
        data_values = [
            no_connection_fraction[i, j],
            forward_only_fraction[i, j],
            reverse_only_fraction[i, j],
            symmetric_fraction[i, j]
        ]
        
        # Get expected (null hypothesis) values
        expected_values = [
            no_connection_null[i, j],
            forward_only_null[i, j],
            reverse_only_null[i, j],
            symmetric_null[i, j]
        ]
        
        # Get counts for statistical testing
        data_counts = [
            no_connection_counts[i, j],
            forward_only_counts[i, j],
            reverse_only_counts[i, j],
            symmetric_counts[i, j]
        ]
        total_pairs = total_pairs_counts[i, j]
        
        # Compute p-values
        p_values = []
        for k, (count, expected_prob) in enumerate(zip(data_counts, expected_values)):
            if total_pairs > 0 and expected_prob > 0 and expected_prob < 1:
                result = binomtest(count, total_pairs, expected_prob, alternative='two-sided')
                p_values.append(result.pvalue)
            else:
                p_values.append(1.0)
        
        # Add small offset to avoid zero values for log scale
        epsilon = 1e-6
        data_values_plot = [max(v, epsilon) for v in data_values]
        expected_values_plot = [max(v, epsilon) for v in expected_values]
        
        # Create grouped bars
        bars1 = ax.bar(x - width/2, data_values_plot, width, label='Data',
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, expected_values_plot, width, label='Expected',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Set log scale
        ax.set_yscale('log')
        
        # Add value labels and significance markers
        for k, (bars, orig_values, pval) in enumerate([(bars1, data_values, p_values), (bars2, expected_values, [1.0]*4)]):
            for bar_idx, (bar, orig_val, p) in enumerate(zip(bars, orig_values, pval)):
                if orig_val > 0.001:
                    height = bar.get_height()
                    if k == 0:  # Only for data bars
                        if p < 0.001:
                            sig_marker = '***'
                        elif p < 0.01:
                            sig_marker = '**'
                        elif p < 0.05:
                            sig_marker = '*'
                        else:
                            sig_marker = ''
                    else:
                        sig_marker = ''
                    
                    label_y_offset = height * 1.1
                    ax.text(bar.get_x() + bar.get_width()/2., label_y_offset,
                           f'{orig_val:.3f}',
                           ha='center', va='bottom', fontsize=9)
                    
                    if k == 0 and sig_marker:
                        asterisk_y_offset = label_y_offset * 0.95
                        ax.text(bar.get_x() + bar.get_width()/2., asterisk_y_offset,
                               sig_marker,
                               ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('Motif Type', fontsize=10)
        ax.set_ylabel('Fraction (log scale)', fontsize=10)
        source_count = e_count if i == 0 else i_count
        target_count = e_count if j == 0 else i_count
        ax.set_title(f'{source_type} ({source_count}) → {target_type} ({target_count})',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(motif_labels, fontsize=9, rotation=45, ha='right')
        all_values = [v for v in data_values + expected_values if v > 0]
        if all_values:
            y_min = min(all_values) * 0.5
            y_max = max(all_values) * 3.5
            ax.set_ylim([y_min, y_max])
        else:
            ax.set_ylim([1e-6, 1.0])
        ax.grid(axis='y', alpha=0.3, which='both')
        ax.legend(fontsize=9, loc='upper right')
        
        pair_idx += 1

plt.suptitle('E/I Pairwise Motif Analysis: Data vs. Expected\n(Both Known: Synapses Between Neurons in Dataset)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(f'{figure_dir}/figure3_pairwise_motifs.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figure_dir}/figure3_pairwise_motifs.png")
plt.close()

# ============================================================================
# CREATE DEGREE STATISTICS BOX PLOTS
# ============================================================================
print("\n=== CREATING DEGREE STATISTICS ===")

# Compute individual neuron statistics
individual_stats = {
    'type': [],
    'out_degree': [],
    'in_degree': [],
    'out_synapses': [],
    'in_synapses': []
}

for i in range(syn_mat_sparse.shape[0]):
    clf_type = matrix_idx_to_clf_type.get(i, 'Unknown')
    if clf_type not in ['E', 'I']:
        continue
    
    row = syn_mat_sparse[i, :]
    col = syn_mat_sparse[:, i]
    individual_stats['type'].append(clf_type)
    individual_stats['out_degree'].append(row.nnz)
    individual_stats['in_degree'].append(col.nnz)
    individual_stats['out_synapses'].append(row.data.sum() if row.nnz > 0 else 0)
    individual_stats['in_synapses'].append(col.data.sum() if col.nnz > 0 else 0)

individual_df = pd.DataFrame(individual_stats)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Out-degree and Out-synapses
out_degree_data = [individual_df[individual_df['type'] == 'E']['out_degree'].values,
                   individual_df[individual_df['type'] == 'I']['out_degree'].values]
out_synapse_data = [individual_df[individual_df['type'] == 'E']['out_synapses'].values,
                    individual_df[individual_df['type'] == 'I']['out_synapses'].values]

bp1_degree = axes[0].boxplot(out_degree_data, positions=[0.8, 1.8], widths=0.15,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(alpha=0.8, linewidth=1.5),
                             medianprops=dict(linewidth=2))
bp1_synapse = axes[0].boxplot(out_synapse_data, positions=[1.2, 2.2], widths=0.15,
                              patch_artist=True, showfliers=False,
                              boxprops=dict(alpha=0.5, linewidth=1.5, linestyle='--'),
                              medianprops=dict(linewidth=2, linestyle='--'))

for patch, color in zip(bp1_degree['boxes'], ['#e74c3c', '#3498db']):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
for patch, color in zip(bp1_synapse['boxes'], ['#e74c3c', '#3498db']):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)

axes[0].set_xticks([1.0, 2.0])
axes[0].set_xticklabels(ei_labels, fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Out-Degree and Out-Synapses by E/I Type', fontsize=14, fontweight='bold')
axes[0].grid(False, axis='y')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# In-degree and In-synapses
in_degree_data = [individual_df[individual_df['type'] == 'E']['in_degree'].values,
                  individual_df[individual_df['type'] == 'I']['in_degree'].values]
in_synapse_data = [individual_df[individual_df['type'] == 'E']['in_synapses'].values,
                   individual_df[individual_df['type'] == 'I']['in_synapses'].values]

bp2_degree = axes[1].boxplot(in_degree_data, positions=[0.8, 1.8], widths=0.15,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(alpha=0.8, linewidth=1.5),
                             medianprops=dict(linewidth=2))
bp2_synapse = axes[1].boxplot(in_synapse_data, positions=[1.2, 2.2], widths=0.15,
                              patch_artist=True, showfliers=False,
                              boxprops=dict(alpha=0.5, linewidth=1.5, linestyle='--'),
                              medianprops=dict(linewidth=2, linestyle='--'))

for patch, color in zip(bp2_degree['boxes'], ['#e74c3c', '#3498db']):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
for patch, color in zip(bp2_synapse['boxes'], ['#e74c3c', '#3498db']):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)

axes[1].set_xticks([1.0, 2.0])
axes[1].set_xticklabels(ei_labels, fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('In-Degree and In-Synapses by E/I Type', fontsize=14, fontweight='bold')
axes[1].grid(False, axis='y')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.8, label='Degree', linewidth=1.5),
    Patch(facecolor='gray', alpha=0.5, label='Synapses', linewidth=1.5, linestyle='--')
]
axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

plt.suptitle('E/I Degree and Synapse Statistics (Both Known)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f'{figure_dir}/figure4_degree_statistics.png', dpi=300, bbox_inches='tight')
print(f"Saved: {figure_dir}/figure4_degree_statistics.png")
plt.close()

print("\n=== ANALYSIS COMPLETE ===")
