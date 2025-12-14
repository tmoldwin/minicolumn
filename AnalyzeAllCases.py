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

def analyze_case(case_name, case_description, neurons_df, syn_mat_sparse, mapping, 
                 external_in_synapses, external_out_synapses, figure_dir):
    """Analyze a specific case and generate all statistics and figures"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {case_name.upper()}")
    print(f"{case_description}")
    print(f"{'='*80}")
    
    os.makedirs(figure_dir, exist_ok=True)
    
    # Create mappings
    root_id_to_cell_type = dict(zip(neurons_df['root_id'], neurons_df['cell_type']))
    root_id_to_clf_type = dict(zip(neurons_df['root_id'], neurons_df['clf_type']))
    root_id_to_idx = {root_id: idx for idx, root_id in mapping.items()}
    
    matrix_idx_to_cell_type = {}
    matrix_idx_to_clf_type = {}
    for idx, root_id in mapping.items():
        matrix_idx_to_cell_type[idx] = root_id_to_cell_type.get(root_id, 'Unknown')
        matrix_idx_to_clf_type[idx] = root_id_to_clf_type.get(root_id, 'Unknown')
    
    # Get cell type counts
    cell_type_counts = Counter(matrix_idx_to_cell_type.values())
    clf_type_counts = Counter(matrix_idx_to_clf_type.values())
    unique_cell_types = sorted(set(matrix_idx_to_cell_type.values()))
    n_cell_types = len(unique_cell_types)
    cell_type_to_idx = {ct: i for i, ct in enumerate(unique_cell_types)}
    
    # Define ordering
    e_neuron_order = ['23P', '4P', '5P-IT', '5P-PT', '5P-NP', '6P-IT', '6P-CT', '6P-U', 'WM-P', 'Unsure E']
    i_neuron_order = ['BC', 'MC', 'BPC', 'NGC', 'Unsure I']
    ordered_cell_types = [ct for ct in e_neuron_order + i_neuron_order if ct in cell_type_counts]
    
    # Initialize matrices based on case
    if case_name == 'both_known':
        # Use connectivity matrix
        cell_type_matrix = np.zeros((n_cell_types, n_cell_types))
        cell_type_synapse_matrix = np.zeros((n_cell_types, n_cell_types))
        
        for i in range(syn_mat_sparse.shape[0]):
            source_cell_type = matrix_idx_to_cell_type.get(i, 'Unknown')
            source_idx = cell_type_to_idx[source_cell_type]
            row = syn_mat_sparse[i, :]
            if row.nnz > 0:
                targets = row.indices
                weights = row.data
                for target_idx, weight in zip(targets, weights):
                    target_cell_type = matrix_idx_to_cell_type.get(target_idx, 'Unknown')
                    target_type_idx = cell_type_to_idx[target_cell_type]
                    cell_type_matrix[source_idx, target_type_idx] += 1
                    cell_type_synapse_matrix[source_idx, target_type_idx] += weight
        
        # Individual neuron stats from matrix
        individual_stats = {
            'cell_type': [],
            'out_degree': [],
            'in_degree': [],
            'out_synapses': [],
            'in_synapses': []
        }
        for i in range(syn_mat_sparse.shape[0]):
            cell_type = matrix_idx_to_cell_type.get(i, 'Unknown')
            row = syn_mat_sparse[i, :]
            col = syn_mat_sparse[:, i]
            individual_stats['cell_type'].append(cell_type)
            individual_stats['out_degree'].append(row.nnz)
            individual_stats['in_degree'].append(col.nnz)
            individual_stats['out_synapses'].append(row.data.sum() if row.nnz > 0 else 0)
            individual_stats['in_synapses'].append(col.data.sum() if col.nnz > 0 else 0)
            
    elif case_name == 'post_known':
        # External inputs: synapses coming from unknown sources
        # These are per-neuron, not a connectivity matrix
        cell_type_matrix = np.zeros((n_cell_types, 1))  # Single column for "external"
        cell_type_synapse_matrix = np.zeros((n_cell_types, 1))
        
        for idx, root_id in mapping.items():
            cell_type = matrix_idx_to_cell_type.get(idx, 'Unknown')
            cell_type_idx = cell_type_to_idx[cell_type]
            ext_synapses = external_in_synapses.get(root_id, 0)
            if ext_synapses > 0:
                cell_type_matrix[cell_type_idx, 0] += 1  # Count neurons with external inputs
                cell_type_synapse_matrix[cell_type_idx, 0] += ext_synapses
        
        # Individual neuron stats
        individual_stats = {
            'cell_type': [],
            'out_degree': [],
            'in_degree': [],
            'out_synapses': [],
            'in_synapses': []
        }
        for idx, root_id in mapping.items():
            cell_type = matrix_idx_to_cell_type.get(idx, 'Unknown')
            ext_synapses = external_in_synapses.get(root_id, 0)
            individual_stats['cell_type'].append(cell_type)
            individual_stats['out_degree'].append(0)  # No outgoing external
            # For external inputs, use synapse count as proxy for "degree" since we don't have connection info
            individual_stats['in_degree'].append(ext_synapses)
            individual_stats['out_synapses'].append(0)
            individual_stats['in_synapses'].append(ext_synapses)
            
    elif case_name == 'pre_known':
        # External outputs: synapses going to unknown targets
        cell_type_matrix = np.zeros((1, n_cell_types))  # Single row for "external"
        cell_type_synapse_matrix = np.zeros((1, n_cell_types))
        
        for idx, root_id in mapping.items():
            cell_type = matrix_idx_to_cell_type.get(idx, 'Unknown')
            cell_type_idx = cell_type_to_idx[cell_type]
            ext_synapses = external_out_synapses.get(root_id, 0)
            if ext_synapses > 0:
                cell_type_matrix[0, cell_type_idx] += 1  # Count neurons with external outputs
                cell_type_synapse_matrix[0, cell_type_idx] += ext_synapses
        
        # Individual neuron stats
        individual_stats = {
            'cell_type': [],
            'out_degree': [],
            'in_degree': [],
            'out_synapses': [],
            'in_synapses': []
        }
        for idx, root_id in mapping.items():
            cell_type = matrix_idx_to_cell_type.get(idx, 'Unknown')
            ext_synapses = external_out_synapses.get(root_id, 0)
            individual_stats['cell_type'].append(cell_type)
            # For external outputs, use synapse count as proxy for "degree" since we don't have connection info
            individual_stats['out_degree'].append(ext_synapses)
            individual_stats['in_degree'].append(0)  # No incoming external
            individual_stats['out_synapses'].append(ext_synapses)
            individual_stats['in_synapses'].append(0)
    
    individual_df = pd.DataFrame(individual_stats)
    
    # Print statistics
    print(f"\n=== CELL TYPE STATISTICS ===")
    print(f"Total neurons: {len(matrix_idx_to_cell_type)}")
    print(f"\nClassification type counts:")
    for clf_type, count in sorted(clf_type_counts.items()):
        print(f"  {clf_type}: {count} ({100*count/len(matrix_idx_to_cell_type):.1f}%)")
    
    print(f"\nCell type counts:")
    for cell_type, count in sorted(cell_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {cell_type}: {count} ({100*count/len(matrix_idx_to_cell_type):.1f}%)")
    
    # Degree statistics
    print(f"\n=== DEGREE STATISTICS BY CELL TYPE ===")
    print(f"{'Cell Type':<20} {'Mean Out-Degree':<20} {'Mean In-Degree':<20} {'Mean Out-Synapses':<20} {'Mean In-Synapses':<20}")
    print("-" * 100)
    
    degree_stats = {
        'cell_types': [],
        'out_degree': [],
        'in_degree': [],
        'out_synapses': [],
        'in_synapses': []
    }
    
    for i, cell_type in enumerate(unique_cell_types):
        count = cell_type_counts[cell_type]
        if case_name == 'both_known':
            out_degree = cell_type_matrix[i, :].sum() / count if count > 0 else 0
            in_degree = cell_type_matrix[:, i].sum() / count if count > 0 else 0
            out_synapses = cell_type_synapse_matrix[i, :].sum() / count if count > 0 else 0
            in_synapses = cell_type_synapse_matrix[:, i].sum() / count if count > 0 else 0
        elif case_name == 'post_known':
            out_degree = 0
            # Mean external input synapses per neuron
            in_degree = cell_type_synapse_matrix[i, 0] / count if count > 0 else 0
            out_synapses = 0
            in_synapses = cell_type_synapse_matrix[i, 0] / count if count > 0 else 0
        elif case_name == 'pre_known':
            # Mean external output synapses per neuron
            out_degree = cell_type_synapse_matrix[0, i] / count if count > 0 else 0
            in_degree = 0
            out_synapses = cell_type_synapse_matrix[0, i] / count if count > 0 else 0
            in_synapses = 0
        
        print(f"{cell_type:<20} {out_degree:<20.2f} {in_degree:<20.2f} {out_synapses:<20.2f} {in_synapses:<20.2f}")
        
        degree_stats['cell_types'].append(cell_type)
        degree_stats['out_degree'].append(out_degree)
        degree_stats['in_degree'].append(in_degree)
        degree_stats['out_synapses'].append(out_synapses)
        degree_stats['in_synapses'].append(in_synapses)
    
    # Summary statistics
    total_connections = cell_type_matrix.sum()
    total_synapses = cell_type_synapse_matrix.sum()
    mean_synapses_per_connection = total_synapses / total_connections if total_connections > 0 else 0
    
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total connections: {total_connections:,.0f}")
    print(f"Total synapses: {total_synapses:,.0f}")
    print(f"Mean synapses per connection: {mean_synapses_per_connection:.2f}")
    
    # Create visualizations
    create_visualizations(case_name, case_description, neurons_df, cell_type_counts, clf_type_counts,
                         cell_type_matrix, cell_type_synapse_matrix, unique_cell_types, ordered_cell_types,
                         degree_stats, individual_df, e_neuron_order, i_neuron_order, figure_dir)
    
    return cell_type_matrix, cell_type_synapse_matrix, degree_stats, individual_df

def create_visualizations(case_name, case_description, neurons_df, cell_type_counts, clf_type_counts,
                         cell_type_matrix, cell_type_synapse_matrix, unique_cell_types, ordered_cell_types,
                         degree_stats, individual_df, e_neuron_order, i_neuron_order, figure_dir):
    """Create all visualizations for a case"""
    
    print(f"\n=== CREATING VISUALIZATIONS ===")
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    colors_ordered = [get_cell_type_color(ct, e_neuron_order, i_neuron_order) for ct in ordered_cell_types]
    cell_type_counts_ordered = [cell_type_counts[ct] for ct in ordered_cell_types]
    
    # Reorder degree stats
    degree_stats_ordered = {
        'cell_types': [],
        'out_degree': [],
        'in_degree': [],
        'out_synapses': [],
        'in_synapses': []
    }
    for ct in ordered_cell_types:
        if ct in degree_stats['cell_types']:
            idx = degree_stats['cell_types'].index(ct)
            degree_stats_ordered['cell_types'].append(ct)
            degree_stats_ordered['out_degree'].append(degree_stats['out_degree'][idx])
            degree_stats_ordered['in_degree'].append(degree_stats['in_degree'][idx])
            degree_stats_ordered['out_synapses'].append(degree_stats['out_synapses'][idx])
            degree_stats_ordered['in_synapses'].append(degree_stats['in_synapses'][idx])
    
    # Figure 1: Cell Type Distribution and E/I Classification
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: Cell type bar chart
    axes[0, 0].barh(range(len(ordered_cell_types)), cell_type_counts_ordered, color=colors_ordered)
    axes[0, 0].set_yticks(range(len(ordered_cell_types)))
    axes[0, 0].set_yticklabels(ordered_cell_types)
    axes[0, 0].set_xlabel('Number of Neurons', fontsize=11)
    axes[0, 0].set_title('Cell Type Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(False)
    
    # Top-right: E/I bar chart
    ei_counts = [clf_type_counts.get('E', 0), clf_type_counts.get('I', 0)]
    ei_labels = ['Excitatory (E)', 'Inhibitory (I)']
    ei_colors = ['#e74c3c', '#3498db']
    axes[0, 1].bar(ei_labels, ei_counts, color=ei_colors, alpha=0.7)
    axes[0, 1].set_ylabel('Number of Neurons', fontsize=11)
    axes[0, 1].set_title('E/I Classification', fontsize=12, fontweight='bold')
    axes[0, 1].grid(False)
    for i, v in enumerate(ei_counts):
        axes[0, 1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Bottom-left: Out-degree
    if case_name != 'post_known':
        axes[1, 0].barh(range(len(degree_stats_ordered['cell_types'])), 
                        degree_stats_ordered['out_degree'], color=colors_ordered, alpha=0.7)
        axes[1, 0].set_yticks(range(len(degree_stats_ordered['cell_types'])))
        axes[1, 0].set_yticklabels(degree_stats_ordered['cell_types'])
        axes[1, 0].set_xlabel('Mean Out-Degree', fontsize=11)
        axes[1, 0].set_title('Mean Out-Degree by Cell Type', fontsize=12, fontweight='bold')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(False)
    else:
        axes[1, 0].axis('off')
    
    # Bottom-right: In-degree
    if case_name != 'pre_known':
        axes[1, 1].barh(range(len(degree_stats_ordered['cell_types'])), 
                        degree_stats_ordered['in_degree'], color=colors_ordered, alpha=0.7)
        axes[1, 1].set_yticks(range(len(degree_stats_ordered['cell_types'])))
        axes[1, 1].set_yticklabels(degree_stats_ordered['cell_types'])
        axes[1, 1].set_xlabel('Mean In-Degree', fontsize=11)
        axes[1, 1].set_title('Mean In-Degree by Cell Type', fontsize=12, fontweight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(False)
    else:
        axes[1, 1].axis('off')
    
    plt.suptitle(f'{case_description}\nCell Type Statistics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(f'{figure_dir}/figure1_cell_types_and_degrees.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {figure_dir}/figure1_cell_types_and_degrees.png")
    plt.close()
    
    # Figure 2: Connectivity Matrix (if applicable)
    if case_name == 'both_known':
        # Reorder matrix
        ordered_to_original_idx = {ct: unique_cell_types.index(ct) for ct in ordered_cell_types}
        cell_type_matrix_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
        cell_type_synapse_matrix_ordered = np.zeros((len(ordered_cell_types), len(ordered_cell_types)))
        
        for i, ct_i in enumerate(ordered_cell_types):
            orig_i = ordered_to_original_idx[ct_i]
            for j, ct_j in enumerate(ordered_cell_types):
                orig_j = ordered_to_original_idx[ct_j]
                cell_type_matrix_ordered[i, j] = cell_type_matrix[orig_i, orig_j]
                cell_type_synapse_matrix_ordered[i, j] = cell_type_synapse_matrix[orig_i, orig_j]
        
        # Calculate probability and mean synapses matrices
        prob_matrix_ordered = np.zeros_like(cell_type_matrix_ordered, dtype=float)
        mean_syn_matrix_ordered = np.zeros_like(cell_type_matrix_ordered, dtype=float)
        
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
                
                n_conn = cell_type_matrix_ordered[i, j]
                n_syn = cell_type_synapse_matrix_ordered[i, j]
                if n_conn > 0:
                    mean_syn_matrix_ordered[i, j] = n_syn / n_conn
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        
        # Number of Connections
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
        
        # Connection Probability
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
        
        # Mean Synapses per Connection
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
        
        # Total Synapses
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
        
        plt.suptitle(f'{case_description}\nCell Type Connectivity Matrices', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(f'{figure_dir}/figure2_connectivity_matrices.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {figure_dir}/figure2_connectivity_matrices.png")
        plt.close()
    
    # Figure 3: Degree and Synapse Statistics (Box Plots)
    if len(individual_df) > 0:
        individual_df_ordered = individual_df[individual_df['cell_type'].isin(ordered_cell_types)].copy()
        individual_df_ordered['cell_type'] = pd.Categorical(individual_df_ordered['cell_type'], 
                                                              categories=ordered_cell_types, ordered=True)
        individual_df_ordered = individual_df_ordered.sort_values('cell_type')
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Prepare data for box plots
        if case_name != 'post_known':
            # Out-degree and Out-synapses
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
                if len(ct_data) > 0 and (ct_data['out_degree'].sum() > 0 or ct_data['out_synapses'].sum() > 0):
                    out_degree_data.append(ct_data['out_degree'].values)
                    out_synapse_data.append(ct_data['out_synapses'].values)
                    out_positions_degree.append(pos - 0.2)
                    out_positions_synapse.append(pos + 0.2)
                    out_labels.append(ct)
                    ct_color = get_cell_type_color(ct, e_neuron_order, i_neuron_order)
                    out_colors_degree.append(ct_color)
                    out_colors_synapse.append(ct_color)
                    pos += 1
            
            if len(out_degree_data) > 0:
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
            else:
                axes[0].axis('off')
        else:
            axes[0].axis('off')
        
        if case_name != 'pre_known':
            # In-degree and In-synapses
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
                if len(ct_data) > 0 and (ct_data['in_degree'].sum() > 0 or ct_data['in_synapses'].sum() > 0):
                    in_degree_data.append(ct_data['in_degree'].values)
                    in_synapse_data.append(ct_data['in_synapses'].values)
                    in_positions_degree.append(pos - 0.2)
                    in_positions_synapse.append(pos + 0.2)
                    in_labels.append(ct)
                    ct_color = get_cell_type_color(ct, e_neuron_order, i_neuron_order)
                    in_colors_degree.append(ct_color)
                    in_colors_synapse.append(ct_color)
                    pos += 1
            
            if len(in_degree_data) > 0:
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
            else:
                axes[1].axis('off')
        else:
            axes[1].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.8, label='Degree', linewidth=1.5),
            Patch(facecolor='gray', alpha=0.5, label='Synapses', linewidth=1.5, linestyle='--')
        ]
        if case_name != 'post_known' and len(out_degree_data) > 0:
            axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        elif case_name != 'pre_known' and len(in_degree_data) > 0:
            axes[1].legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.suptitle(f'{case_description}\nDegree and Synapse Statistics', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(f'{figure_dir}/figure3_degree_statistics.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {figure_dir}/figure3_degree_statistics.png")
        plt.close()

# Main execution
if __name__ == '__main__':
    print("Loading data...")
    CONNECTIVITY_DIR = '.'
    syn_mat_sparse, mapping = load_connectivity(CONNECTIVITY_DIR, 'network_synapses')
    
    neurons_df = pd.read_csv('connectome_neurons.csv')
    print(f"Loaded {len(neurons_df)} neurons")
    
    # Calculate external synapses
    root_id_to_idx = {root_id: idx for idx, root_id in mapping.items()}
    neurons_df['matrix_idx'] = neurons_df['root_id'].map(root_id_to_idx)
    
    # External inputs: ds_num_of_incoming_synapses - num_of_incoming_synapses
    neurons_df['external_in_synapses'] = neurons_df['ds_num_of_incoming_synapses'] - neurons_df['num_of_incoming_synapses']
    external_in_synapses = dict(zip(neurons_df['root_id'], neurons_df['external_in_synapses']))
    
    # External outputs: ds_num_of_outgoing_synapses - num_of_outgoing_synapses
    neurons_df['external_out_synapses'] = neurons_df['ds_num_of_outgoing_synapses'] - neurons_df['num_of_outgoing_synapses']
    external_out_synapses = dict(zip(neurons_df['root_id'], neurons_df['external_out_synapses']))
    
    print(f"\nExternal inputs total: {neurons_df['external_in_synapses'].sum():,.0f}")
    print(f"External outputs total: {neurons_df['external_out_synapses'].sum():,.0f}")
    
    # Analyze all three cases
    cases = [
        ('both_known', 'Both Known: Synapses Between Neurons in Dataset', 'figures/both_known'),
        ('post_known', 'Post-Known: External Inputs (Unknown Sources)', 'figures/post_known'),
        ('pre_known', 'Pre-Known: External Outputs (Unknown Targets)', 'figures/pre_known')
    ]
    
    for case_name, case_description, figure_dir in cases:
        analyze_case(case_name, case_description, neurons_df, syn_mat_sparse, mapping,
                     external_in_synapses, external_out_synapses, figure_dir)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)

