import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_hist(data, xlabel, ylabel, title, bins=50):
    mean_val = data.mean()
    std_val = data.std()

    # Calculate thresholds
    std_neg_1 = mean_val - std_val
    std_neg_2 = mean_val - 2 * std_val

    std_pos_1 = mean_val + std_val
    std_pos_2 = mean_val + 2 * std_val

    # Count outliers
    num_below_2std = (data < std_neg_2).sum()
    num_above_2std = (data > std_pos_2).sum()

    # Plot
    plt.figure(figsize=(12, 6))
    sns.histplot(data, bins=bins, kde=False, color='skyblue')

    # Vertical lines for stds
    plt.axvline(mean_val, color='black', linestyle='--', label=f'Mean = {mean_val:.2f}')

    plt.axvline(std_neg_1, color='orange', linestyle='--', label=f'-1 STD = {std_neg_1:.2f}')
    plt.axvline(std_neg_2, color='red', linestyle='--', label=f'-2 STD = {std_neg_2:.2f}')

    plt.axvline(std_pos_1, color='orange', linestyle='--', label=f'+1 STD = {std_pos_1:.2f}')
    plt.axvline(std_pos_2, color='red', linestyle='--', label=f'+2 STD = {std_pos_2:.2f}')

    # Annotate counts beyond ±3 STD
    ymax = plt.ylim()[1]
    plt.text(std_neg_2, ymax * 0.9, f'num counts < -2 STD: {num_below_2std}', color='purple', ha='right')
    plt.text(std_pos_2, ymax * 0.9, f'num counts > +2 STD: {num_above_2std}', color='purple', ha='left')

    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_categorical_distribution(adata, column, top_n=None, figsize=(8, 4)):
    """Plot bar chart of counts per category in a given obs column."""
    if column not in adata.obs:
        print(f"Column '{column}' not found in adata.obs.")
        return

    counts = adata.obs[column].value_counts().head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(x=list(counts.values), y=list(counts.index))
    plt.xlabel('Number of cells')
    plt.ylabel(column)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.grid(axis='x')
    plt.show()   
    
def plot_heatmap(data, xlabel, ylabel, title, figsize=(16, 16)):
    plt.figure(figsize=figsize)
    sns.heatmap(data, cmap='viridis', annot=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    
def plot_bar_comaprision(adata, measure='cell_type',group='assay'):
    pd.crosstab(adata.obs[group], adata.obs[measure], normalize='index').plot(kind='bar', stacked=True,figsize=(16,8))

    # Build a cross-tabulation: assay × cell_type
    cross_data = pd.crosstab(adata.obs[measure], adata.obs[group])

    # Count in how many assays each cell type appears
    assay_presence = (cross_data > 0).sum(axis=1)

    # Filter cell types that appear in only one assay
    exclusive_celltypes = assay_presence[assay_presence == 1].index.tolist()

    # Show them
    if len(exclusive_celltypes) > 0:
        print(f"{measure} exclusive to one {group}:")
        for ct in exclusive_celltypes:
            groups = cross_data.columns[cross_data.loc[ct] > 0].tolist()
            print(f" - {ct} (only in {group}: {groups[0]})")
