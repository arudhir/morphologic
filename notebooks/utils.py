import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from sklearn.feature_selection import mutual_info_classif
from scipy.sparse import issparse
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import featurewiz
from harmonypy import run_harmony

def get_crispr_perturbations(adata):
    adata = adata.copy()
    return adata[adata.obs['perturbation_type'] == 'CRISPR']

def run_pca(adata, n_components=50, standardize=True, copy=False, key_added='X_pca'):
    if copy:
        adata = adata.copy()
    X = adata.X.copy()
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    adata.obsm[key_added] = X_pca
    adata.uns['pca'] = {
        'variance_ratio': pca.explained_variance_ratio_,
        'variance': pca.explained_variance_,
        'total_variance': sum(pca.explained_variance_ratio_)
    }
    return adata

def run_umap(adata, n_components=2, copy=False, key_added='X_umap'):
    if copy:
        adata = adata.copy()
    reducer = UMAP(n_components=n_components)
    adata.obsm[key_added] = reducer.fit_transform(adata.obsm['X_pca'])
    return adata

def plot_dim_reduction(adata, basis='X_pca', color_by='treatment', style_by=None, dims=[0, 1], 
             title='PCA of Cell Profiles', palette=None, cmap=None):
    """
    plot PCA results with options to color and style points based on metadata.
    
    Parameters:
    - adata: AnnData object
    - basis: str, key in `adata.obsm` where PCA coordinates are stored
    - color_by: str, column in `adata.obs` to use for coloring points
    - style_by: str or None, column in `adata.obs` to use for styling points
    - dims: list, dimensions of PCA to plot (e.g., [0, 1] for PC1 vs PC2)
    - title: str, title of the plot
    - palette: dict or list, color palette for categorical `color_by` data
    - cmap: str or colormap, colormap for continuous `color_by` data
    """
    # get the pca coordinates
    X_pca = adata.obsm[basis]
    
    # get explained variance
    variance_ratio = adata.uns['pca']['variance_ratio']
    
    # axis labels with explained variance
    x_label = f"PC{dims[0] + 1} ({variance_ratio[dims[0]] * 100:.2f}%)"
    y_label = f"PC{dims[1] + 1} ({variance_ratio[dims[1]] * 100:.2f}%)"
    
    # set up plot kwargs
    plot_kwargs = {
        'x': X_pca[:, dims[0]], 
        'y': X_pca[:, dims[1]], 
        'hue': adata.obs[color_by],
    }
    
    # add style if provided
    if style_by:
        plot_kwargs['style'] = adata.obs[style_by]
    
    # add palette or cmap based on data type of `color_by`
    if palette:
        plot_kwargs['palette'] = palette
    elif cmap and adata.obs[color_by].dtype.kind in 'if':  # continuous data
        plot_kwargs['palette'] = cmap
    
    # plot
    scatterplot = sns.scatterplot(**plot_kwargs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    
    # move legend to the right off the main plot
    scatterplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=color_by)
    
    plt.show()

def clean_adata(adata, nan_threshold=0.5, standardize=True):
    """
    Clean AnnData by handling inf/nan, removing rows/columns with too many NaNs, and standardizing features.

    Parameters:
        adata (AnnData): Input AnnData object to clean.
        nan_threshold (float): Maximum fraction of NaNs allowed in a row to keep it. Defaults to 0.5.
        standardize (bool): Whether to standardize the columns. Defaults to True.

    Returns:
        AnnData: A cleaned AnnData object.
    """
    import numpy as np
    import anndata as ad

    X_clean = adata.X.copy()

    # replace inf with nan
    X_clean = np.where(np.isinf(X_clean), np.nan, X_clean)
    
    # identify rows/columns with all NaNs
    row_mask = ~np.all(np.isnan(X_clean), axis=1)
    col_mask = ~np.all(np.isnan(X_clean), axis=0)
    
    # filter rows/columns with at least one valid value
    X_clean = X_clean[row_mask][:, col_mask]
    
    # drop rows with NaN fraction above the threshold
    row_nan_fraction = np.isnan(X_clean).mean(axis=1)
    valid_rows = row_nan_fraction < nan_threshold
    X_clean = X_clean[valid_rows]

    # standardize columns (mean 0, variance 1)
    if standardize:
        for col in range(X_clean.shape[1]):
            mean = np.nanmean(X_clean[:, col])
            std = np.nanstd(X_clean[:, col])
            if std > 0:  # avoid division by zero
                X_clean[:, col] = (X_clean[:, col] - mean) / std
    
    # create new AnnData object with cleaned data
    adata_clean = ad.AnnData(
        X=X_clean, 
        obs=adata.obs[row_mask][valid_rows].copy(),  # ensure indices align
        var=adata.var[col_mask].copy()
    )
    
    # store cleaning metadata
    adata_clean.uns['cleaning_info'] = {
        'kept_row_indices': np.where(row_mask)[0][valid_rows],
        'kept_col_indices': np.where(col_mask)[0],
        'n_rows_dropped': len(row_mask) - np.sum(valid_rows),
        'n_cols_dropped': len(col_mask) - np.sum(col_mask)
    }
    
    return adata_clean

def select_features(adata, variance_threshold=0.1, correlation_threshold=0.9, target_col=None):
    """
    Select features based on variance and correlation thresholds.
    
    Parameters:
        adata (AnnData): The input AnnData object.
        variance_threshold (float): Threshold for minimum variance. Defaults to 0.1.
        correlation_threshold (float): Threshold for maximum allowed correlation between features. Defaults to 0.9.
        target_col (str, optional): Column in adata.obs to use as target for supervised feature selection.
    
    Returns:
        selected_features (list): List of selected feature names.
        stats (DataFrame): DataFrame containing variance and correlation information.
    """
    import numpy as np
    import pandas as pd

    # Compute variances and apply variance filter
    variances = np.var(adata.X, axis=0)
    valid_variance_mask = variances > variance_threshold

    # Exclude features with NaNs
    valid_nan_mask = ~np.isnan(adata.X).any(axis=0)

    # Combine masks
    valid_features_mask = valid_variance_mask & valid_nan_mask

    # Subset the data to valid features
    X_valid = adata.X[:, valid_features_mask]
    feature_names_valid = np.array(adata.var_names)[valid_features_mask]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_valid.T)

    # Zero out the diagonal to ignore self-correlations
    np.fill_diagonal(corr_matrix, 0)

    # Identify pairs of features with high correlation
    correlated_pairs = np.argwhere(np.abs(corr_matrix) > correlation_threshold)

    # Keep track of features to remove
    features_to_remove = set()

    # Iterate over correlated pairs and decide which feature to remove
    for idx1, idx2 in correlated_pairs:
        # Remove the feature with lower variance
        if variances[idx1] < variances[idx2]:
            features_to_remove.add(idx1)
        else:
            features_to_remove.add(idx2)

    # Create mask for features to keep
    features_to_keep = [i for i in range(len(feature_names_valid)) if i not in features_to_remove]

    # Select the features
    selected_features = feature_names_valid[features_to_keep]

    # Prepare stats DataFrame
    stats = pd.DataFrame({
        'feature_name': feature_names_valid,
        'variance': variances[valid_features_mask],
        'correlated': [i in features_to_remove for i in range(len(feature_names_valid))]
    })

    return selected_features.tolist(), stats

def print_feature_selection_stats(stats):
    """Print feature selection statistics in a readable format."""
    print("\nFeature Selection Statistics:")
    print("-" * 50)
    print(f"Initial features: {stats['initial_features']}")
    print(f"After variance filtering: {stats['after_variance']} "
          f"({stats['after_variance']/stats['initial_features']*100:.1f}%)")
    print(f"After correlation filtering: {stats['after_correlation']} "
          f"({stats['after_correlation']/stats['initial_features']*100:.1f}%)")
    print(f"Final features: {stats['final_features']} "
          f"({stats['final_features']/stats['initial_features']*100:.1f}%)")
    
    if 'top_features' in stats:
        print("\nTop 10 features by mutual information:")
        print("-" * 50)
        for i, (feature, importance) in enumerate(stats['top_features'], 1):
            print(f"{i}. {feature}: {importance:.4f}")

def plot_correlation_heatmap(adata, obs_var, threshold=0.9):
    """
    Plot a heatmap of feature correlations with colorbars for metadata.

    Parameters:
    - adata: AnnData object with feature matrix and metadata.
    - obs_var: str, column in `adata.obs` to color rows/columns by.
    - threshold: Correlation value to highlight highly correlated features.
    """
    # convert to pandas DataFrame
    df = pd.DataFrame(
        adata.X.toarray() if issparse(adata.X) else adata.X,
        columns=adata.var_names
    )
    
    # compute correlation matrix
    corr_matrix = df.corr()

    # create metadata color mapping
    if obs_var not in adata.obs.columns:
        raise ValueError(f"'{obs_var}' not found in adata.obs.")

    # map metadata to colors
    metadata = pd.Categorical(adata.obs[obs_var])
    palette = sns.color_palette('tab10', len(metadata.categories))
    metadata_colors = metadata.map(dict(zip(metadata.categories, palette)))

    # create colorbars for heatmap
    row_colors = metadata_colors.values
    col_colors = metadata_colors.values

    # mask the upper triangle for clarity
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # plot
    plt.figure(figsize=(12, 10))
    sns.clustermap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        row_colors=row_colors,
        col_colors=col_colors,
        mask=mask,
        figsize=(12, 10),
        cbar_kws={'label': 'Correlation'},
    )
    plt.title('Correlation Heatmap with Metadata')
    plt.show()

def combat_correct(adata, batch_key, covariates=None):
    """
    Apply ComBat batch correction with checks.
    """

    # Ensure no NaNs in X
    if np.isnan(adata.X).any():
        adata.X = np.nan_to_num(adata.X, nan=0.0)

    # Ensure no invalid batch info
    if batch_key not in adata.obs or adata.obs[batch_key].isnull().any():
        raise ValueError(f"'{batch_key}' column contains missing or invalid batch information.")
    
    # Apply ComBat
    sc.pp.combat(adata, key=batch_key, covariates=covariates, inplace=True)

def compute_r2_scores(adata, batch_key):
    """
    Compute R² scores for each principal component (PC) to quantify variance explained by batch.

    Parameters:
        adata (AnnData): Input AnnData object.
        batch_key (str): Column in adata.obs to use as batch label.

    Returns:
        list: R² scores for each PC.
    """
    # Create one-hot encoded batch labels
    batch_labels = pd.get_dummies(adata.obs[batch_key])

    # Compute R² scores for each PC
    r2_scores = []
    for pc in adata.obsm["X_pca"].T:  # Transpose to iterate over PCs
        model = LinearRegression()
        model.fit(batch_labels, pc)
        r2_scores.append(r2_score(pc, model.predict(batch_labels)))
    return r2_scores

def plot_r2_scores(r2_scores_before, r2_scores_after):
    """
    Plot R² scores and cumulative R² scores before and after batch correction.

    Parameters:
        r2_scores_before (list): R² scores before batch correction.
        r2_scores_after (list): R² scores after batch correction.
    """
    pcs = np.arange(1, len(r2_scores_before) + 1)

    # Plot R² scores
    plt.figure(figsize=(10, 6))
    plt.plot(pcs, r2_scores_before, label="Before ComBat", marker="o")
    plt.plot(pcs, r2_scores_after, label="After ComBat", marker="o")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained by Batch (R²)")
    plt.title("Variance Explained by Batch Before and After ComBat")
    plt.legend()
    plt.show()

    # Plot cumulative R² scores
    plt.figure(figsize=(10, 6))
    plt.plot(pcs, np.cumsum(r2_scores_before), label="Cumulative Before ComBat", marker="o")
    plt.plot(pcs, np.cumsum(r2_scores_after), label="Cumulative After ComBat", marker="o")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Variance Explained by Batch (R²)")
    plt.title("Cumulative Variance Explained Before and After ComBat")
    plt.legend()
    plt.show()

def plot_histograms_by_group(adata, var_names, obs_names, bins=20, legend_loc="upper right"):
    """
    Plot histograms for specified features (var_names) stratified by sample groups (obs_names).

    Parameters:
        adata (AnnData): Input AnnData object.
        var_names (list of str): List of feature names to plot.
        obs_names (str): Column name in adata.obs to stratify the histogram by.
        bins (int): Number of bins for the histograms. Defaults to 20.
        legend_loc (str): Location for the legend. Defaults to "upper right".

    Returns:
        None
    """
    if adata.is_view:
        adata = adata.copy()

    if obs_names not in adata.obs.columns:
        raise ValueError(f"'{obs_names}' not found in adata.obs.")

    # Ensure stratification column is categorical
    adata.obs[obs_names] = adata.obs[obs_names].fillna("Unknown").astype("category")

    for var_name in var_names:
        if var_name not in adata.var_names:
            print(f"Warning: '{var_name}' not found in adata.var_names. Skipping.")
            continue

        # Extract data
        data = pd.DataFrame({
            obs_names: adata.obs[obs_names],
            var_name: adata[:, var_name].X.flatten()
        })

        # Plot histogram stratified by group
        plt.figure(figsize=(8, 6))
        histplot = sns.histplot(
            data=data,
            x=var_name,
            hue=obs_names,
            bins=bins,
            kde=True,
            stat="count",
            multiple="stack",
            alpha=0.3
        )

        # Explicitly fetch and set legend
        handles, labels = histplot.get_legend_handles_labels()
        if handles and labels:
            plt.legend(handles, labels, title=obs_names, loc=legend_loc)
        else:
            print(f"Warning: No legend found for '{var_name}' stratified by '{obs_names}'.")

        # Add labels and title
        plt.title(f"Histogram of '{var_name}' Stratified by '{obs_names}'")
        plt.xlabel(var_name)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

def plot_boxplots_by_group(adata, var_names, obs_names):
    """
    Plot boxplots for specified features (var_names) stratified by sample groups (obs_names).

    Parameters:
        adata (AnnData): Input AnnData object.
        var_names (list of str): List of feature names to plot.
        obs_names (str): Column name in adata.obs to stratify the boxplot by.

    Returns:
        None
    """
    if adata.is_view:
        adata = adata.copy()

    if obs_names not in adata.obs.columns:
        raise ValueError(f"'{obs_names}' not found in adata.obs.")

    # Ensure stratification column is categorical
    adata.obs[obs_names] = adata.obs[obs_names].fillna("Unknown").astype("category")

    for var_name in var_names:
        if var_name not in adata.var_names:
            print(f"Warning: '{var_name}' not found in adata.var_names. Skipping.")
            continue

        # Extract data
        data = pd.DataFrame({
            obs_names: adata.obs[obs_names],
            var_name: adata[:, var_name].X.flatten()
        })

        # Plot boxplot
        plt.figure(figsize=(10, 6))
        boxplot = sns.boxplot(
            data=data,
            x=obs_names,
            y=var_name,
            showfliers=False,
            palette="Set3"
        )
        sns.stripplot(
            data=data,
            x=obs_names,
            y=var_name,
            color="black",
            alpha=0.4,
            jitter=True,
            size=3
        )

        # Add labels and title
        plt.title(f"Boxplot of '{var_name}' Stratified by '{obs_names}'")
        plt.xlabel(obs_names)
        plt.ylabel(var_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def plot_dose_response(
    adata,
    var_name,
    x_axis,
    stratify_by,
    log_scale=True,
    figsize=(12, 6),
    col_wrap=3,
    compound_empty_control_ids=None,
    egfrko_empty_control_ids=None,
    egfrko_nonempty_control_ids=None,
    control_colors=None,
):
    """
    create a dose-response visualization: mean ± variance ribbons across concentrations,
    with optional horizontal lines for control conditions.

    parameters:
        adata (annData): input AnnData object containing inhibitor data.
        var_name (str): feature name to plot.
        x_axis (str): column name in adata.obs for the x-axis (e.g., concentration).
        stratify_by (str): column name in adata.obs to stratify treatments (e.g., treatment).
        log_scale (bool): whether to use a log scale for the x-axis. defaults to True.
        figsize (tuple): size of the entire figure. defaults to (12, 6).
        col_wrap (int): number of columns for subplots in the facet grid. defaults to 3.
        compound_empty_control_ids (index, optional): obs_names for compound empty control wells.
        egfrko_empty_control_ids (index, optional): obs_names for egfr ko empty control wells.
        egfrko_nonempty_control_ids (index, optional): obs_names for egfr ko non-empty control wells.
        control_colors (dict, optional): dictionary mapping control types to colors.

    returns:
        none
    """
    if x_axis not in adata.obs.columns:
        raise ValueError(f"'{x_axis}' not found in adata.obs.")
    if stratify_by not in adata.obs.columns:
        raise ValueError(f"'{stratify_by}' not found in adata.obs.")
    if var_name not in adata.var_names:
        raise ValueError(f"'{var_name}' not found in adata.var_names.")

    # filter out invalid rows
    valid_obs_mask = ~adata.obs[[x_axis, stratify_by]].isnull().any(axis=1)
    filtered_adata = adata[valid_obs_mask].copy()

    # create a dataframe for seaborn
    data = pd.DataFrame({
        x_axis: filtered_adata.obs[x_axis],
        stratify_by: filtered_adata.obs[stratify_by],
        var_name: filtered_adata[:, var_name].X.flatten(),
    })

    # aggregate means and variances
    summary = (
        data.groupby([stratify_by, x_axis])[var_name]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "Mean", "std": "Std"})
    )

    # compute means for control conditions
    control_means = {}
    if compound_empty_control_ids is not None:
        control_means["inhibitor_control"] = np.nanmean(adata[compound_empty_control_ids, var_name].X.ravel())
    if egfrko_empty_control_ids is not None:
        control_means["egfrko_control"] = np.nanmean(adata[egfrko_empty_control_ids, var_name].X.ravel())
    if egfrko_nonempty_control_ids is not None:
        control_means["egfrko"] = np.nanmean(adata[egfrko_nonempty_control_ids, var_name].X.ravel())

    # default control colors
    if not control_colors:
        control_colors = {
            "inhibitor_control": "red",
            "egfrko_control": "green",
            "egfrko": "purple",
        }

    # initialize facet grid
    g = sns.FacetGrid(
        summary, col=stratify_by, col_wrap=col_wrap, height=4, sharex=True, sharey=True
    )

    # plot mean ± variance ribbons
    def plot_ribbon(data, **kwargs):
        x = data[x_axis]
        y = data["Mean"]
        yerr = data["Std"]
        ax = plt.gca()
        ax.plot(x, y, marker="o", linestyle="-", color="tab:blue")
        ax.fill_between(x, y - yerr, y + yerr, color="tab:blue", alpha=0.2)

        # plot horizontal lines for controls
        for control, mean_value in control_means.items():
            ax.axhline(
                mean_value, color=control_colors[control], linestyle="--", alpha=0.8, label=control
            )

    g.map_dataframe(plot_ribbon)

    # optional: log scale
    if log_scale:
        for ax in g.axes.flatten():
            ax.set_xscale("log")

    # customize plot
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels(x_axis, var_name)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Dose-Response for '{var_name}'")

    # add a single legend
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    g.fig.legend(
        handles, labels, title="Controls", loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3
    )
    plt.show()

class NaNFilter(BaseEstimator, TransformerMixin):
    """
    Remove columns with too many NaN values and replace remaining NaNs or inf/-inf with 0.
    """
    def __init__(self, nan_threshold=0.5):
        self.nan_threshold = nan_threshold
        self.valid_columns_ = None

    def fit(self, X, y=None):
        # Treat inf/-inf as NaN for filtering
        X = np.where(np.isinf(X), np.nan, X)

        # Compute fraction of NaNs per column
        nan_fraction = np.isnan(X).mean(axis=0)
        self.valid_columns_ = nan_fraction < self.nan_threshold
        return self

    def transform(self, X, y=None):
        # Treat inf/-inf as NaN for transformation
        X = np.where(np.isinf(X), np.nan, X)

        # Select valid columns and fill remaining NaNs with 0
        X_filtered = X[:, self.valid_columns_]
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        return imputer.fit_transform(X_filtered)

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove highly correlated features.
    """
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_keep_ = None

    def fit(self, X, y=None):
        # handle both numpy arrays and pandas dataframes
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # compute the correlation matrix
        corr_matrix = X.corr().abs()
        # identify features to keep
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        self.to_keep_ = ~corr_matrix.where(upper_triangle).apply(
            lambda x: any(x > self.threshold), axis=0
        )
        return self

    def transform(self, X, y=None):
        # select features to keep
        if isinstance(X, np.ndarray):
            return X[:, self.to_keep_.values]
        return X.loc[:, self.to_keep_]


def create_feature_selection_pipeline(
    var_threshold=0.01, corr_threshold=0.9, nan_threshold=0.7, k=100
):
    """
    Create a sklearn pipeline for feature selection.

    Parameters:
        var_threshold (float): Threshold for variance filtering.
        corr_threshold (float): Threshold for correlation filtering.
        nan_threshold (float): Maximum fraction of NaNs allowed per column.
        k (int): Number of features to select with SelectKBest.

    Returns:
        Pipeline: A sklearn pipeline object.
    """
    return Pipeline([
        ("nan_filter", NaNFilter(nan_threshold=nan_threshold)),
        ("variance_filter", VarianceThreshold(threshold=var_threshold)),
        ("correlation_filter", CorrelationFilter(threshold=corr_threshold)),
        ("select_k_best", SelectKBest(score_func=f_classif, k=k))
    ])


def apply_feature_selection_pipeline(adata, pipeline, target):
    """
    Apply a sklearn feature selection pipeline to an AnnData object.

    Parameters:
        adata (AnnData): Input AnnData object.
        pipeline (Pipeline): Sklearn feature selection pipeline.
        target (str): Column name in adata.obs containing target labels.

    Returns:
        AnnData: New AnnData object with selected features.
    """
    # ensure target column exists
    if target not in adata.obs.columns:
        raise ValueError(f"Target column '{target}' not found in adata.obs.")

    # extract X (features) and y (labels)
    X = adata.X
    y = adata.obs[target].values

    # fit the pipeline and transform X
    X_selected = pipeline.fit_transform(X, y)

    # handle case where select_k_best might retain fewer features than requested
    if hasattr(pipeline.named_steps["select_k_best"], "get_support"):
        selected_features = pipeline.named_steps["select_k_best"].get_support(indices=True)
    else:
        selected_features = np.arange(X_selected.shape[1])

    # update adata.var to match selected features
    selected_var = adata.var.iloc[selected_features].copy()

    # create a new AnnData object with the selected features
    adata_selected = adata[:, selected_features].copy()

    # update X and var
    adata_selected.X = X_selected
    adata_selected.var = selected_var

    return adata_selected

def inspect_features(adata):
    """
    Generate a summary table with variance, correlation, and NaN fraction for each feature.

    Parameters:
        adata (AnnData): Input AnnData object.

    Returns:
        pd.DataFrame: Table summarizing feature-level statistics.
    """
    # compute variance
    feature_variance = np.nanvar(adata.X, axis=0)

    # compute NaN fraction
    nan_fraction = np.isnan(adata.X).mean(axis=0)

    # compute absolute correlations
    corr_matrix = pd.DataFrame(adata.X).corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    max_corr = corr_matrix.where(upper_triangle).max()

    # combine all metrics into a dataframe
    feature_summary = pd.DataFrame({
        "feature_name": adata.var_names,
        "variance": feature_variance,
        "nan_fraction": nan_fraction,
        "max_correlation": max_corr.values
    })

    # sort by nan_fraction and variance for easier inspection
    feature_summary = feature_summary.sort_values(by=["nan_fraction", "variance"], ascending=[True, False])

    return feature_summary.reset_index(drop=True)

def filter_var_names_by_pattern(adata, remove_terms):
    """
    Filter out var_names in an AnnData object that match any of the specified patterns.

    Parameters:
        adata (AnnData): Input AnnData object.
        remove_terms (list): List of substrings to exclude from var_names.

    Returns:
        AnnData: A new AnnData object with filtered var_names.
    """
    import re

    # Create regex pattern from the list of terms
    pattern = '|'.join(re.escape(term) for term in remove_terms)
    
    # Find matching var_names
    matching_vars = adata.var_names[adata.var_names.str.contains(pattern, na=False)]
    
    # Print the number of features removed
    print(f"Number of features removed: {len(matching_vars)}")
    
    # Filter the AnnData object
    filtered_adata = adata[:, ~adata.var_names.isin(matching_vars)].copy()
    
    return filtered_adata

def plot_boxplots_by_group_facet(adata, var_name, group_by, stratify_by, col_wrap=3):
    """
    Create subplots for boxplots of a specific feature across groups, stratified by another variable.

    Parameters:
        adata (AnnData): Input AnnData object.
        var_name (str): Feature name to plot.
        group_by (str): Column name in adata.obs to create subplots for each unique value.
        stratify_by (str): Column name in adata.obs to stratify the boxplot within each subplot.
        col_wrap (int): Number of columns for subplot grid. Defaults to 2.

    Returns:
        None
    """
    # ensure the grouping column exists
    if group_by not in adata.obs.columns:
        raise ValueError(f"'{group_by}' not found in adata.obs.")

    # ensure the stratification column exists
    if stratify_by not in adata.obs.columns:
        raise ValueError(f"'{stratify_by}' not found in adata.obs.")

    # ensure the feature exists
    if var_name not in adata.var_names:
        raise ValueError(f"'{var_name}' not found in adata.var_names.")

    # create a dataframe with necessary data
    data = pd.DataFrame({
        group_by: adata.obs[group_by],
        stratify_by: adata.obs[stratify_by],
        var_name: adata[:, var_name].X.flatten(),
    })

    # create a facet grid for boxplots
    g = sns.FacetGrid(data, col=group_by, col_wrap=col_wrap, height=4, sharey=True)
    g.map_dataframe(
        sns.boxplot,
        x=stratify_by,
        y=var_name,
        palette="Set3",
        showfliers=False,
    )
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels(stratify_by, var_name)
    g.fig.suptitle(f"Boxplots of '{var_name}' stratified by '{stratify_by}'", y=1.05)
    g.tight_layout()
    plt.show()

from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import StandardScaler

def run_tsne(adata, n_components=2, perplexity=30, learning_rate=200, standardize=True, copy=False, key_added='X_tsne'):
    """
    Run t-SNE on an AnnData object.

    Parameters:
        adata (AnnData): Input AnnData object.
        n_components (int): Number of dimensions for t-SNE. Defaults to 2.
        perplexity (float): t-SNE perplexity. Defaults to 30.
        learning_rate (float): t-SNE learning rate. Defaults to 200.
        standardize (bool): Whether to standardize the input data before t-SNE. Defaults to True.
        copy (bool): Whether to return a copy of the AnnData object. Defaults to False.
        key_added (str): Key under which to save the t-SNE results in `adata.obsm`. Defaults to 'X_tsne'.

    Returns:
        AnnData: AnnData object with t-SNE results added.
    """
    if copy:
        adata = adata.copy()

    # Handle potential NaNs or infinities
    X = adata.X.copy()
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Standardize if required
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Run t-SNE
    tsne = TSNE(
        n_components=n_components, 
        perplexity=perplexity, 
        learning_rate=learning_rate, 
        random_state=42
    )
    X_tsne = tsne.fit_transform(X)

    # Store t-SNE results in AnnData
    adata.obsm[key_added] = X_tsne
    adata.uns['tsne'] = {
        'perplexity': perplexity,
        'learning_rate': learning_rate,
        'n_components': n_components
    }
    return adata

class FeatureWizTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to integrate FeatureWiz into a sklearn pipeline.

    Parameters:
        target (str): Target column name for featurewiz.
        corr_limit (float): Correlation limit for featurewiz. Defaults to 0.9.
        verbose (int): Verbosity level for featurewiz. Defaults to 1.
    """
    def __init__(self, target, corr_limit=0.9, verbose=0):
        self.target = target
        self.corr_limit = corr_limit
        self.verbose = verbose
        self.selected_features_ = None

    def fit(self, X, y=None):
        """
        Fit the FeatureWizTransformer to the data.

        Parameters:
            X (pd.DataFrame): Feature data.
            y (pd.Series or array-like): Target data.

        Returns:
            self: Fitted transformer.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # Convert to DataFrame if X is not already one
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)

        if y is not None:
            df[self.target] = y
        else:
            raise ValueError("Target values (y) must be provided for FeatureWiz.")

        # Run FeatureWiz
        self.selected_features_, _ = featurewiz(
            dataname=df,
            target=self.target,
            corr_limit=self.corr_limit,
            verbose=0
        )
        return self

    def transform(self, X):
        """
        Transform the data using selected features from FeatureWiz.

        Parameters:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("The transformer has not been fitted yet. Call fit() before transform().")

        # Select only the features chosen by FeatureWiz
        return X[self.selected_features_]

    def get_support(self, indices=False):
        """
        Get the indices or names of selected features.

        Parameters:
            indices (bool): Whether to return indices instead of names.

        Returns:
            list: Selected feature names or indices.
        """
        if self.selected_features_ is None:
            raise ValueError("The transformer has not been fitted yet. Call fit() before get_support().")

        if indices:
            return [i for i, col in enumerate(self.selected_features_)]
        return self.selected_features_

def create_featurewiz_pipeline(target_col, corr_limit=0.9, verbose=1):
    """
    Create a pipeline with FeatureWiz for feature selection.

    Parameters:
        target_col (str): Column name for target variable.
        corr_limit (float): Correlation limit for FeatureWiz. Defaults to 0.9.
        verbose (int): Verbosity level for FeatureWiz. Defaults to 1.

    Returns:
        Pipeline: A sklearn pipeline with FeatureWiz.
    """
    return Pipeline([
        ("featurewiz", FeatureWizTransformer(target=target_col, corr_limit=corr_limit, verbose=verbose))
        ])

def run_harmony_batch_correction(adata, batch_key="batch", vars_use=None, key_added="X_harmony"):
    """
    Perform Harmony batch correction on an AnnData object.

    Parameters:
        adata (AnnData): The input AnnData object.
        batch_key (str): The column in adata.obs specifying batch information.
        vars_use (list): List of covariates from adata.obs to use in Harmony correction.
        key_added (str): Key under adata.obsm to store the corrected matrix.

    Returns:
        AnnData: The input AnnData object with corrected embeddings stored in adata.obsm[key_added].
    """
    adata = adata.copy()
    if vars_use is None:
        vars_use = []

    # Run Harmony batch correction
    harmony_result = run_harmony(
        np.nan_to_num(adata.X),  # Expression matrix with NaNs handled
        meta_data=adata.obs,     # Metadata (dataframe)
        vars_use=vars_use + [batch_key]  # Covariates + batch key
    )

    # Extract corrected embeddings
    corrected = harmony_result.Z_corr.T  # Transpose to match AnnData's convention
    adata.obsm[key_added] = corrected

    return adata

