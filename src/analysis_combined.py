import pandas as pd
import sys
import pickle as original_pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean

def load_pandas_pickle(filepath):
    """Load pandas pickle with compatibility for old versions"""
    
    if 'pandas.core.indexes.numeric' not in sys.modules:
        import pandas as pd
        
        class NumericIndexModule:
            Float64Index = pd.Index
            Int64Index = pd.Index
            
            def __getattr__(self, name):
                if hasattr(pd, name):
                    return getattr(pd, name)
                elif hasattr(pd.core.indexes.base, name):
                    return getattr(pd.core.indexes.base, name)
                elif name.endswith('Index'):
                    return pd.Index
                else:
                    raise AttributeError(f"module has no attribute '{name}'")
        
        sys.modules['pandas.core.indexes.numeric'] = NumericIndexModule()
    
    with open(filepath, 'rb') as f:
        return original_pickle.load(f)


def load_rule_based_results(dataset, name, num_runs=None):
    """Load raw results from rule-based models (TR, BRS, etc.)"""
    try:
        rs = load_pandas_pickle(f'results/{dataset}/{name}_rs.pkl')
        
        # Limit to specified number of runs if provided
        if num_runs is not None:
            for cost in rs.index:
                for col in rs.columns:
                    rs.loc[cost, col] = rs.loc[cost, col][:num_runs]
        
        # Recalculate means and stderrs from rs
        means = rs.apply(lambda x: x.apply(lambda y: mean(y)))
        stderrs = rs.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))
        
        return means, stderrs, rs
    except FileNotFoundError:
        print(f"Rule-based results not found for {dataset}/{name}")
        return None, None, None


def load_xgb_results(dataset, name, num_runs=None):
    """Load raw results from XGBoost models"""
    try:
        rs = load_pandas_pickle(f'results/{dataset}/{name}_xgb_comprehensive_rs.pkl')
        
        # Limit to specified number of runs if provided
        if num_runs is not None:
            for cost in rs.index:
                for col in rs.columns:
                    rs.loc[cost, col] = rs.loc[cost, col][:num_runs]
        
        # Recalculate means and stderrs from rs
        means = rs.apply(lambda x: x.apply(lambda y: mean(y)))
        stderrs = rs.apply(lambda x: x.apply(lambda y: np.std(y)/np.sqrt(len(y))))
        
        return means, stderrs, rs
    except FileNotFoundError:
        print(f"XGBoost results not found for {dataset}/{name}")
        return None, None, None


def combine_results(rule_means, rule_std, xgb_means, xgb_std):
    """Combine rule-based and XGBoost results into a single dataframe"""
    
    # Use whichever is available as base
    if rule_means is not None:
        combined_means = rule_means.copy()
        combined_std = rule_std.copy()
    elif xgb_means is not None:
        combined_means = xgb_means.copy()
        combined_std = xgb_std.copy()
    else:
        return None, None
    
    # Add the other results
    if rule_means is not None and xgb_means is not None:
        for col in xgb_means.columns:
            combined_means[col] = xgb_means[col]
            combined_std[col] = xgb_std[col]
    
    return combined_means, combined_std


def normalize_to_human_baseline(results_means, results_stderrs, normalize=True):
    """
    Normalize all metrics relative to human decision loss (value added)
    If normalize=False, just returns the original results
    """
    if not normalize:
        return results_means, results_stderrs
    
    normalized_means = results_means.copy()
    normalized_stderrs = results_stderrs.copy()
    
    # Subtract all metrics from human_decision_loss to show improvement
    for col in results_means.columns:
        if col != 'human_decision_loss':
            normalized_means[col] = results_means['human_decision_loss'] - results_means[col]
    
    # Human baseline becomes 0
    normalized_means['human_decision_loss'] = 0
    
    return normalized_means, normalized_stderrs


def make_combined_plot(results_means, results_stderrs, name, ax, 
                       methods_to_plot, stopat=6, set_x=False, set_y=False):
    """
    Create a combined plot with specified methods
    
    methods_to_plot: dict with keys as method names and values as column names
    Example: {
        'TR': 'tr_team_w_reset_objective',
        'Base_XGB_Raw': 'base_xgb_raw_final_objective',
    }
    """
    
    color_dict = {
        'TR': '#348ABD', 
        'TR-no(Cost)': '#CC79A7',
        'TR-no(ADB)': '#8EBA42',
        'TR-no(ADB, Cost)': '#E24A33',
        'Task-Only (Current Practice)': '#988ED5',
        'Human': 'darkgray', 
        'Task-Only XGB': '#FF6B6B', 
        'Task-Only XGB Filtered': '#FF9999',
        'RevAI XGB': '#4ECDC4', 
        'RevAI XGB Filtered': '#7DDDD7',
    }
    
    marker_dict = {
        'TR': '.', 
        'TR-no(Cost)': '^',
        'TR-no(ADB)': 'x',
        'TR-no(ADB, Cost)': 'v',
        'Task-Only (Current Practice)': 's',
        'Human': '-', 
        'Task-Only XGB': 'o', 
        'Task-Only XGB Filtered': 's',
        'RevAI XGB': '^', 
        'RevAI XGB Filtered': 'd',
    }

        # Dictionary to control visibility - True means invisible (alpha=0)
    invisible_dict = {
        'TR': True,
        'TR-no(Cost)': False,
        'TR-no(ADB)': False,
        'TR-no(ADB, Cost)': False,
        'Task-Only (Current Practice)': False,
        'Human': False,
        'Task-Only XGB': True,  # Set to True to make invisible
        'Task-Only XGB Filtered': True,  # Set to True to make invisible
        'RevAI XGB': True,  # Set to True to make invisible
        'RevAI XGB Filtered': True,  # Set to True to make invisible
    }

    
    # Plot each method
    for method_name, column_name in methods_to_plot.items():
        if column_name not in results_means.columns:
            print(f"Warning: Column '{column_name}' not found in results")
            continue

        # Determine alpha based on visibility
        is_invisible = invisible_dict.get(method_name, False)
        line_alpha = 0 if is_invisible else (0.5 if method_name == 'Human' else 1.0)
        fill_alpha = 0 if is_invisible else 0.22
            
        if method_name == 'Human':
            # Human baseline is plotted differently
            ax.plot(results_means.index[0:stopat], 
                   results_means[column_name].iloc[0:stopat], 
                   c=color_dict.get(method_name, 'gray'), 
                   markersize=1, 
                   label=method_name, 
                   ls='--', 
                   alpha=line_alpha)
        else:
            ax.plot(results_means.index[0:stopat], 
                   results_means[column_name].iloc[0:stopat], 
                   marker=marker_dict.get(method_name, 'o'), 
                   c=color_dict.get(method_name, 'gray'), 
                   label=method_name, 
                   markersize=2.1, 
                   linewidth=1,
                   alpha=line_alpha)
        
        # Add error bands
        ax.fill_between(results_means.index[0:stopat], 
                       results_means[column_name].iloc[0:stopat] - 
                       results_stderrs[column_name].iloc[0:stopat],
                       results_means[column_name].iloc[0:stopat] + 
                       results_stderrs[column_name].iloc[0:stopat],
                       color=color_dict.get(method_name, 'gray'), 
                       alpha=fill_alpha)
    
    if set_x:
        ax.set_xlabel('Reconciliation Cost', fontsize=7)
    if set_y:
        ax.set_ylabel('Value Added vs Human Alone', fontsize=7)
    ax.tick_params(labelrotation=45, labelsize=6)
    ax.grid('on', linestyle='dotted', linewidth=0.2, color='black')
    
    # Add horizontal line at y=0 for human baseline (if normalized)
    if 'human_decision_loss' in results_means.columns and results_means['human_decision_loss'].iloc[0] == 0:
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    return ax


# ============================================================================
# CONFIGURATION SECTION - Edit this to control what gets plotted
# ============================================================================

# Define which methods to plot
# Set method name as key, and column name as value
# Set to None or comment out to disable
METHODS_TO_PLOT = {
    # Rule-based methods
    'TR': 'tr_team_w_reset_objective',
    'TR-no(Cost)': 'tr2s_team_w_reset_objective',
    'TR-no(ADB)': 'trnoadb_team_w_reset_objective',
    'TR-no(ADB, Cost)': 'hyrs_norecon_objective',
    'Task-Only (Current Practice)': 'brs_team_objective',
    'Human': 'human_decision_loss',
    
    # XGBoost methods - uncomment to enable
    #'Task-Only XGB': 'base_xgb_raw_final_objective',
    #'Task-Only XGB Filtered': 'base_xgb_filtered_final_objective',
    #'RevAI XGB': 'synth_xgb_raw_final_objective',
    #'RevAI XGB': 'synth_xgb_filtered_final_objective',
}

# Filter out None values
METHODS_TO_PLOT = {k: v for k, v in METHODS_TO_PLOT.items() if v is not None}
# Number of runs to use (set to None to use all available)
NUM_RULE_RUNS = 10  # Use first 10 runs from rule-based results
NUM_XGB_RUNS = 10    # Use all 10 runs from XGBoost results
# Normalize results to show value added vs human baseline
NORMALIZE_TO_HUMAN = True
# Plotting parameters
costs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
stopat = 6  # Only plot first 6 cost values

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    datasets = ['heart_disease', 'fico', 'hr']
    names = ['biased', 'biased_dec_bias', 'offset_01']
    
    # Create subplot grid with more spacing
    fig, axs = plt.subplots(3, 3, figsize=(6, 5.5))  # Slightly taller figure
    
    cols = ['Difficulty-biased decisions \n Group-biased ADB',
            'Group-biased decisions \n Group-biased ADB',
            'Difficulty-biased decisions \n Accuracy-biased ADB']
    rows = ['Heart Disease', ' FICO ', ' HR ']
    subs = ['a','b','c','d','e','f','g','h','i']
    
    pad = 5
    datarow = 0
    behaviorrow = 0
    
    for dataset in datasets:
        for name in names:
            print(f"\nProcessing {dataset}/{name}...")
            
            # Load rule-based results
            rule_means, rule_std, rule_rs = load_rule_based_results(
                dataset, name, num_runs=NUM_RULE_RUNS
            )
            
            # Load XGBoost results
            xgb_means, xgb_std, xgb_rs = load_xgb_results(
                dataset, name, num_runs=NUM_XGB_RUNS
            )
            
            # Combine results
            combined_means, combined_std = combine_results(
                rule_means, rule_std, xgb_means, xgb_std
            )
            
            if combined_means is None:
                print(f"No results found for {dataset}/{name}")
                behaviorrow += 1
                continue
            
            # Normalize if requested
            combined_means, combined_std = normalize_to_human_baseline(
                combined_means, combined_std, normalize=NORMALIZE_TO_HUMAN
            )
            
            # Determine axis labels
            set_y = (behaviorrow == 0)
            set_x = (dataset == 'hr')
            
            # Create plot
            ax = make_combined_plot(
                combined_means, combined_std, name, axs[datarow, behaviorrow],
                METHODS_TO_PLOT, stopat=stopat, set_x=set_x, set_y=set_y
            )
            
            behaviorrow += 1
        
        datarow += 1
        behaviorrow = 0
    
    # Add column headers
    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=8, ha='center', va='baseline')
    
    # Add row labels closer to the plots
    for ax, row in zip(axs[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=8, ha='right', va='center', rotation=90)
    
    # Add subplot labels - moved further right to avoid overlap with y-axis
    for ax, sub in zip(axs.flatten(), subs):
        ax.annotate(sub, xy=(-0.15, 0.95), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=10, ha='center', va='baseline', weight='bold')
    
    # Add legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7)
    
    # Adjust layout with more spacing
    fig.tight_layout()
    plt.subplots_adjust(
        bottom=0.15,    # Space for legend
        hspace=0.35,    # Increased vertical spacing between rows
        wspace=0.35,    # Increased horizontal spacing for y-axis labels
        left=0.12,      # Enough space for row labels
        right=0.98      # Keep right edge tight
    )
    
    fig.savefig(f'combined_results_humantask5_presentation.pdf', format='pdf')
    
    print("\nPlot saved as 'combined_results.jpg'")
    print(f"Used {NUM_RULE_RUNS} runs for rule-based methods")
    print(f"Used {NUM_XGB_RUNS} runs for XGBoost methods")
