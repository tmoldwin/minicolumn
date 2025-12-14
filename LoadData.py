import pickle
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, spmatrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_connectivity(path: str, name: str) -> tuple[spmatrix, dict]:
    sparse_matrix = load_npz(f'{path}/{name}_matrix.npz')
    with open(f'{path}/{name}_mapping.pkl', 'rb') as fp:
        mapping = pickle.load(fp)
    return sparse_matrix, mapping

CONNECTIVITY_DIR = '.'
syn_mat_sparse, mapping = load_connectivity(CONNECTIVITY_DIR, 'network_synapses')
bin_mat_sparse = syn_mat_sparse.T.copy() # make sure to Transpose it!
bin_mat_sparse[bin_mat_sparse > 1] = 1

# Load neuron metadata
neurons_df = pd.read_csv('connectome_neurons.csv')
print(f"Loaded {len(neurons_df)} neurons")

# Create mapping from root_id to cell type
root_id_to_cell_type = dict(zip(neurons_df['root_id'], neurons_df['cell_type']))
root_id_to_clf_type = dict(zip(neurons_df['root_id'], neurons_df['clf_type']))

# Map matrix indices to cell types
matrix_idx_to_cell_type = {}
matrix_idx_to_clf_type = {}
for idx, root_id in mapping.items():
    matrix_idx_to_cell_type[idx] = root_id_to_cell_type.get(root_id, 'Unknown')
    matrix_idx_to_clf_type[idx] = root_id_to_clf_type.get(root_id, 'Unknown')

# Basic cell type statistics
print("\n=== CELL TYPE STATISTICS ===")
cell_type_counts = Counter(matrix_idx_to_cell_type.values())
clf_type_counts = Counter(matrix_idx_to_clf_type.values())

print(f"\nTotal neurons: {len(matrix_idx_to_cell_type)}")
print(f"\nClassification type counts:")
for clf_type, count in sorted(clf_type_counts.items()):
    print(f"  {clf_type}: {count} ({100*count/len(matrix_idx_to_cell_type):.1f}%)")

print(f"\nCell type counts:")
for cell_type, count in sorted(cell_type_counts.items(), key=lambda x: -x[1]):
    print(f"  {cell_type}: {count} ({100*count/len(matrix_idx_to_cell_type):.1f}%)")

# Create cell type connectivity matrix
unique_cell_types = sorted(set(matrix_idx_to_cell_type.values()))
n_cell_types = len(unique_cell_types)
cell_type_matrix = np.zeros((n_cell_types, n_cell_types))
cell_type_synapse_matrix = np.zeros((n_cell_types, n_cell_types))

# Map cell types to indices
cell_type_to_idx = {ct: i for i, ct in enumerate(unique_cell_types)}

# Aggregate connections by cell type
print("\n=== COMPUTING CELL TYPE CONNECTIVITY MATRIX ===")
# Use syn_mat_sparse directly (not transposed) - row i connects to column j
for i in range(syn_mat_sparse.shape[0]):
    source_cell_type = matrix_idx_to_cell_type.get(i, 'Unknown')
    source_idx = cell_type_to_idx[source_cell_type]
    
    # Get outgoing connections (row i -> columns j)
    row = syn_mat_sparse[i, :]
    if row.nnz > 0:
        targets = row.indices
        weights = row.data
        for target_idx, weight in zip(targets, weights):
            target_cell_type = matrix_idx_to_cell_type.get(target_idx, 'Unknown')
            target_type_idx = cell_type_to_idx[target_cell_type]
            
            # Binary connectivity
            cell_type_matrix[source_idx, target_type_idx] += 1
            
            # Synapse count
            cell_type_synapse_matrix[source_idx, target_type_idx] += weight

print(f"\nCell type connectivity matrix shape: {cell_type_matrix.shape}")

# Pairwise connection statistics
print("\n=== PAIRWISE CONNECTION STATISTICS ===")
print("\nBinary connectivity (number of connections):")
print(f"{'Source':<20} {'Target':<20} {'Connections':<15} {'% of source':<15} {'% of target':<15}")
print("-" * 85)

for i, source_type in enumerate(unique_cell_types):
    for j, target_type in enumerate(unique_cell_types):
        n_connections = int(cell_type_matrix[i, j])
        if n_connections > 0:
            source_count = cell_type_counts[source_type]
            target_count = cell_type_counts[target_type]
            pct_source = 100 * n_connections / source_count if source_count > 0 else 0
            pct_target = 100 * n_connections / target_count if target_count > 0 else 0
            print(f"{source_type:<20} {target_type:<20} {n_connections:<15} {pct_source:<15.2f} {pct_target:<15.2f}")

print("\nSynapse counts:")
print(f"{'Source':<20} {'Target':<20} {'Total Synapses':<20} {'Mean per connection':<20}")
print("-" * 80)

for i, source_type in enumerate(unique_cell_types):
    for j, target_type in enumerate(unique_cell_types):
        n_synapses = cell_type_synapse_matrix[i, j]
        n_connections = cell_type_matrix[i, j]
        if n_connections > 0:
            mean_synapses = n_synapses / n_connections
            print(f"{source_type:<20} {target_type:<20} {n_synapses:<20.0f} {mean_synapses:<20.2f}")

# Connection probability statistics
print("\n=== CONNECTION PROBABILITY STATISTICS ===")
print(f"{'Source':<20} {'Target':<20} {'Connection Prob':<20}")
print("-" * 60)

for i, source_type in enumerate(unique_cell_types):
    for j, target_type in enumerate(unique_cell_types):
        n_connections = cell_type_matrix[i, j]
        source_count = cell_type_counts[source_type]
        target_count = cell_type_counts[target_type]
        if source_count > 0 and target_count > 0:
            # Connection probability = connections / (source_count * target_count)
            # For self-connections, divide by source_count * (target_count - 1)
            if i == j:
                possible_connections = source_count * (target_count - 1)
            else:
                possible_connections = source_count * target_count
            
            if possible_connections > 0:
                conn_prob = n_connections / possible_connections
                print(f"{source_type:<20} {target_type:<20} {conn_prob:<20.4f}")

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
total_connections = int(cell_type_matrix.sum())
total_synapses = cell_type_synapse_matrix.sum()
mean_synapses_per_connection = total_synapses / total_connections if total_connections > 0 else 0

print(f"Total connections: {total_connections:,}")
print(f"Total synapses: {total_synapses:,.0f}")
print(f"Mean synapses per connection: {mean_synapses_per_connection:.2f}")

# Compute individual neuron statistics for box plots
print("\n=== COMPUTING INDIVIDUAL NEURON STATISTICS ===")
individual_stats = {
    'cell_type': [],
    'out_degree': [],
    'in_degree': [],
    'out_synapses': [],
    'in_synapses': []
}

for i in range(syn_mat_sparse.shape[0]):
    cell_type = matrix_idx_to_cell_type.get(i, 'Unknown')
    
    # Out-degree and out-synapses
    row = syn_mat_sparse[i, :]
    out_degree = row.nnz
    out_synapses = row.data.sum() if row.nnz > 0 else 0
    
    # In-degree and in-synapses
    col = syn_mat_sparse[:, i]
    in_degree = col.nnz
    in_synapses = col.data.sum() if col.nnz > 0 else 0
    
    individual_stats['cell_type'].append(cell_type)
    individual_stats['out_degree'].append(out_degree)
    individual_stats['in_degree'].append(in_degree)
    individual_stats['out_synapses'].append(out_synapses)
    individual_stats['in_synapses'].append(in_synapses)

# Convert to DataFrame for easier manipulation
individual_df = pd.DataFrame(individual_stats)

# In-degree and out-degree by cell type
print("\n=== DEGREE STATISTICS BY CELL TYPE ===")
print(f"{'Cell Type':<20} {'Mean Out-Degree':<20} {'Mean In-Degree':<20} {'Mean Out-Synapses':<20} {'Mean In-Synapses':<20}")
print("-" * 100)

# Store degree statistics for visualization (mean values)
degree_stats = {
    'cell_types': [],
    'out_degree': [],
    'in_degree': [],
    'out_synapses': [],
    'in_synapses': []
}

for i, cell_type in enumerate(unique_cell_types):
    count = cell_type_counts[cell_type]
    out_degree = cell_type_matrix[i, :].sum() / count if count > 0 else 0
    in_degree = cell_type_matrix[:, i].sum() / count if count > 0 else 0
    out_synapses = cell_type_synapse_matrix[i, :].sum() / count if count > 0 else 0
    in_synapses = cell_type_synapse_matrix[:, i].sum() / count if count > 0 else 0
    print(f"{cell_type:<20} {out_degree:<20.2f} {in_degree:<20.2f} {out_synapses:<20.2f} {in_synapses:<20.2f}")
    
    degree_stats['cell_types'].append(cell_type)
    degree_stats['out_degree'].append(out_degree)
    degree_stats['in_degree'].append(in_degree)
    degree_stats['out_synapses'].append(out_synapses)
    degree_stats['in_synapses'].append(in_synapses)

# Create visualizations
print("\n=== CREATING VISUALIZATIONS ===")
import os
os.makedirs('figures', exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Define layer ordering for E neurons (by typical layer, then by name)
# E neurons ordered by layer: L2-L4 (23P, 4P), L5 (5P-IT, 5P-PT, 5P-NP), L6 (6P-IT, 6P-CT, 6P-U), WM (WM-P), Unsure
e_neuron_order = ['23P', '4P', '5P-IT', '5P-PT', '5P-NP', '6P-IT', '6P-CT', '6P-U', 'WM-P', 'Unsure E']
# I neurons grouped together
i_neuron_order = ['BC', 'MC', 'BPC', 'NGC', 'Unsure I']

# Create ordered cell type list: E neurons first (by layer), then I neurons
ordered_cell_types = []
for ct in e_neuron_order:
    if ct in cell_type_counts:
        ordered_cell_types.append(ct)
for ct in i_neuron_order:
    if ct in cell_type_counts:
        ordered_cell_types.append(ct)

# Color scheme: E=red (#e74c3c), I=blue (#3498db)
def get_cell_type_color(cell_type):
    if cell_type in i_neuron_order:
        return '#3498db'  # Blue for inhibitory
    else:
        return '#e74c3c'  # Red for excitatory

colors_ordered = [get_cell_type_color(ct) for ct in ordered_cell_types]
cell_type_counts_ordered = [cell_type_counts[ct] for ct in ordered_cell_types]

# Reorder matrices and degree stats to match ordered_cell_types
ordered_to_original_idx = {ct: unique_cell_types.index(ct) for ct in ordered_cell_types}
cell_type_matrix_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
cell_type_synapse_matrix_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
prob_matrix_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
mean_syn_matrix_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))

for i, ct_i in enumerate(ordered_cell_types):
    orig_i = ordered_to_original_idx[ct_i]
    for j, ct_j in enumerate(ordered_cell_types):
        orig_j = ordered_to_original_idx[ct_j]
        cell_type_matrix_ordered[i, j] = cell_type_matrix[orig_i, orig_j]
        cell_type_synapse_matrix_ordered[i, j] = cell_type_synapse_matrix[orig_i, orig_j]

# Reorder degree stats
degree_stats_ordered = {
    'cell_types': [],
    'out_degree': [],
    'in_degree': [],
    'out_synapses': [],
    'in_synapses': []
}
for ct in ordered_cell_types:
    idx = degree_stats['cell_types'].index(ct)
    degree_stats_ordered['cell_types'].append(ct)
    degree_stats_ordered['out_degree'].append(degree_stats['out_degree'][idx])
    degree_stats_ordered['in_degree'].append(degree_stats['in_degree'][idx])
    degree_stats_ordered['out_synapses'].append(degree_stats['out_synapses'][idx])
    degree_stats_ordered['in_synapses'].append(degree_stats['in_synapses'][idx])

# Calculate probability matrix (using ordered matrix)
for i, source_type in enumerate(ordered_cell_types):
    for j, target_type in enumerate(ordered_cell_types):
        n_connections = cell_type_matrix_ordered[i, j]
        source_count = cell_type_counts[source_type]
        target_count = cell_type_counts[target_type]
        if source_count > 0 and target_count > 0:
            if i == j:
                possible_connections = source_count * (target_count - 1)
            else:
                possible_connections = source_count * target_count
            if possible_connections > 0:
                prob_matrix_ordered[i, j] = n_connections / possible_connections

# Calculate mean synapses matrix (using ordered matrix)
for i in range(len(ordered_cell_types)):
    for j in range(len(ordered_cell_types)):
        n_conn = cell_type_matrix_ordered[i, j]
        n_syn = cell_type_synapse_matrix_ordered[i, j]
        if n_conn > 0:
            mean_syn_matrix_ordered[i, j] = n_syn / n_conn

# Figure 1: Cell Type Distribution and E/I Classification
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: Cell type bar chart (ordered)
axes[0, 0].barh(range(len(ordered_cell_types)), cell_type_counts_ordered, color=colors_ordered)
axes[0, 0].set_yticks(range(len(ordered_cell_types)))
axes[0, 0].set_yticklabels(ordered_cell_types)
axes[0, 0].set_xlabel('Number of Neurons', fontsize=11)
axes[0, 0].set_title('Cell Type Distribution', fontsize=12, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# Top-right: E/I bar chart
ei_counts = [clf_type_counts.get('E', 0), clf_type_counts.get('I', 0)]
ei_labels = ['Excitatory (E)', 'Inhibitory (I)']
ei_colors = ['#e74c3c', '#3498db']  # E=red, I=blue
axes[0, 1].bar(ei_labels, ei_counts, color=ei_colors, alpha=0.7)
axes[0, 1].set_ylabel('Number of Neurons', fontsize=11)
axes[0, 1].set_title('E/I Classification', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(ei_counts):
    axes[0, 1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# Bottom-left: Out-degree
axes[1, 0].barh(range(len(degree_stats_ordered['cell_types'])), 
                degree_stats_ordered['out_degree'], color=colors_ordered, alpha=0.7)
axes[1, 0].set_yticks(range(len(degree_stats_ordered['cell_types'])))
axes[1, 0].set_yticklabels(degree_stats_ordered['cell_types'])
axes[1, 0].set_xlabel('Mean Out-Degree', fontsize=11)
axes[1, 0].set_title('Mean Out-Degree by Cell Type', fontsize=12, fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# Bottom-right: In-degree
axes[1, 1].barh(range(len(degree_stats_ordered['cell_types'])), 
                degree_stats_ordered['in_degree'], color=colors_ordered, alpha=0.7)
axes[1, 1].set_yticks(range(len(degree_stats_ordered['cell_types'])))
axes[1, 1].set_yticklabels(degree_stats_ordered['cell_types'])
axes[1, 1].set_xlabel('Mean In-Degree', fontsize=11)
axes[1, 1].set_title('Mean In-Degree by Cell Type', fontsize=12, fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(axis='x', alpha=0.3)

plt.suptitle('Cell Type Statistics and Connectivity Overview', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('figures/figure1_cell_types_and_degrees.png', dpi=300, bbox_inches='tight')
print("Saved: figures/figure1_cell_types_and_degrees.png")
plt.close()

# Figure 2: Connectivity Matrices (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(20, 18))

# Top-left: Number of Connections (log scale)
conn_matrix_log = np.log10(cell_type_matrix_ordered + 1)
mask1 = cell_type_matrix_ordered == 0
annot_matrix = cell_type_matrix_ordered.astype(int)
sns.heatmap(conn_matrix_log, annot=annot_matrix, fmt='d', cmap='YlOrRd', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask1, cbar_kws={'label': 'Log10(Connections + 1)'}, ax=axes[0, 0],
            linewidths=0.5, linecolor='gray')
axes[0, 0].set_title('Number of Connections (log scale)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Target Cell Type', fontsize=10)
axes[0, 0].set_ylabel('Source Cell Type', fontsize=10)
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=8)
axes[0, 0].tick_params(axis='y', rotation=0, labelsize=8)

# Top-right: Connection Probability
mask2 = prob_matrix_ordered == 0
sns.heatmap(prob_matrix_ordered, annot=True, fmt='.3f', cmap='viridis', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask2, cbar_kws={'label': 'Connection Probability'}, ax=axes[0, 1],
            linewidths=0.5, linecolor='gray')
axes[0, 1].set_title('Connection Probability', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Target Cell Type', fontsize=10)
axes[0, 1].set_ylabel('Source Cell Type', fontsize=10)
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
axes[0, 1].tick_params(axis='y', rotation=0, labelsize=8)

# Bottom-left: Mean Synapses per Connection
mask3 = mean_syn_matrix_ordered == 0
sns.heatmap(mean_syn_matrix_ordered, annot=True, fmt='.2f', cmap='plasma', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask3, cbar_kws={'label': 'Mean Synapses per Connection'}, ax=axes[1, 0],
            linewidths=0.5, linecolor='gray')
axes[1, 0].set_title('Mean Synapses per Connection', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Target Cell Type', fontsize=10)
axes[1, 0].set_ylabel('Source Cell Type', fontsize=10)
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=8)
axes[1, 0].tick_params(axis='y', rotation=0, labelsize=8)

# Bottom-right: Total Synapses (log scale)
syn_matrix_log = np.log10(cell_type_synapse_matrix_ordered + 1)
mask4 = cell_type_synapse_matrix_ordered == 0
annot_syn = cell_type_synapse_matrix_ordered.astype(int)
sns.heatmap(syn_matrix_log, annot=annot_syn, fmt='d', cmap='coolwarm', 
            xticklabels=ordered_cell_types, yticklabels=ordered_cell_types,
            mask=mask4, cbar_kws={'label': 'Log10(Total Synapses + 1)'}, ax=axes[1, 1],
            linewidths=0.5, linecolor='gray')
axes[1, 1].set_title('Total Synapses (log scale)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Target Cell Type', fontsize=10)
axes[1, 1].set_ylabel('Source Cell Type', fontsize=10)
axes[1, 1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1, 1].tick_params(axis='y', rotation=0, labelsize=8)

plt.suptitle('Cell Type Connectivity Matrices', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('figures/figure2_connectivity_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: figures/figure2_connectivity_matrices.png")
plt.close()

# Figure 3: Degree and Synapse Statistics (Box Plots)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Filter individual_df to only include ordered cell types
individual_df_ordered = individual_df[individual_df['cell_type'].isin(ordered_cell_types)].copy()
individual_df_ordered['cell_type'] = pd.Categorical(individual_df_ordered['cell_type'], 
                                                      categories=ordered_cell_types, ordered=True)
individual_df_ordered = individual_df_ordered.sort_values('cell_type')

# Prepare data for grouped box plots
# Left plot: Out-degree and Out-synapses
out_degree_data = []
out_synapse_data = []
out_positions_degree = []
out_positions_synapse = []
out_labels = []
out_colors_degree = []
out_colors_synapse = []

pos = 1
for ct in ordered_cell_types:
    ct_data = individual_df_ordered[individual_df_ordered['cell_type'] == ct]
    if len(ct_data) > 0:
        out_degree_data.append(ct_data['out_degree'].values)
        out_synapse_data.append(ct_data['out_synapses'].values)
        out_positions_degree.append(pos - 0.2)
        out_positions_synapse.append(pos + 0.2)
        out_labels.append(ct)
        ct_color = get_cell_type_color(ct)
        out_colors_degree.append(ct_color)
        # Slightly lighter/darker for synapses to differentiate
        out_colors_synapse.append(ct_color)
        pos += 1

# Create box plots for out-degree and out-synapses
bp1_degree = axes[0].boxplot(out_degree_data, positions=out_positions_degree, widths=0.3,
                             patch_artist=True, showfliers=False, 
                             boxprops=dict(alpha=0.8, linewidth=1.5),
                             medianprops=dict(linewidth=2))
bp1_synapse = axes[0].boxplot(out_synapse_data, positions=out_positions_synapse, widths=0.3,
                              patch_artist=True, showfliers=False,
                              boxprops=dict(alpha=0.5, linewidth=1.5, linestyle='--'),
                              medianprops=dict(linewidth=2, linestyle='--'))

for patch, color in zip(bp1_degree['boxes'], out_colors_degree):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
for patch, color in zip(bp1_synapse['boxes'], out_colors_synapse):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)

axes[0].set_xticks(range(1, len(out_labels) + 1))
axes[0].set_xticklabels(out_labels, rotation=45, ha='right', fontsize=10)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Out-Degree and Out-Synapses by Cell Type', fontsize=14, fontweight='bold')
axes[0].grid(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Right plot: In-degree and In-synapses
in_degree_data = []
in_synapse_data = []
in_positions_degree = []
in_positions_synapse = []
in_labels = []
in_colors_degree = []
in_colors_synapse = []

pos = 1
for ct in ordered_cell_types:
    ct_data = individual_df_ordered[individual_df_ordered['cell_type'] == ct]
    if len(ct_data) > 0:
        in_degree_data.append(ct_data['in_degree'].values)
        in_synapse_data.append(ct_data['in_synapses'].values)
        in_positions_degree.append(pos - 0.2)
        in_positions_synapse.append(pos + 0.2)
        in_labels.append(ct)
        ct_color = get_cell_type_color(ct)
        in_colors_degree.append(ct_color)
        in_colors_synapse.append(ct_color)
        pos += 1

# Create box plots for in-degree and in-synapses
bp2_degree = axes[1].boxplot(in_degree_data, positions=in_positions_degree, widths=0.3,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(alpha=0.8, linewidth=1.5),
                             medianprops=dict(linewidth=2))
bp2_synapse = axes[1].boxplot(in_synapse_data, positions=in_positions_synapse, widths=0.3,
                              patch_artist=True, showfliers=False,
                              boxprops=dict(alpha=0.5, linewidth=1.5, linestyle='--'),
                              medianprops=dict(linewidth=2, linestyle='--'))

for patch, color in zip(bp2_degree['boxes'], in_colors_degree):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)
for patch, color in zip(bp2_synapse['boxes'], in_colors_synapse):
    patch.set_facecolor(color)
    patch.set_edgecolor(color)

axes[1].set_xticks(range(1, len(in_labels) + 1))
axes[1].set_xticklabels(in_labels, rotation=45, ha='right', fontsize=10)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('In-Degree and In-Synapses by Cell Type', fontsize=14, fontweight='bold')
axes[1].grid(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.8, label='Degree', linewidth=1.5),
    Patch(facecolor='gray', alpha=0.5, label='Synapses', linewidth=1.5, linestyle='--')
]
axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

plt.suptitle('Degree and Synapse Statistics by Cell Type', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('figures/figure3_degree_statistics.png', dpi=300, bbox_inches='tight')
print("Saved: figures/figure3_degree_statistics.png")
plt.close()

print("\nAll visualizations saved successfully in 'figures' directory!")