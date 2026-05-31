import numpy as np
import scipy.stats as stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pandas as pd

def weighted_cohen_kappa(y_true, y_pred, weights='quadratic'):
    """
    Calculates Weighted Cohen's Kappa.
    """
    return cohen_kappa_score(y_true, y_pred, weights=weights)

def bland_altman_stats(data1, data2):
    """
    Computes stats for Bland-Altman plot.
    Returns mean_bias, lower_limit, upper_limit.
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    diff = data1 - data2
    mean_bias = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    upper_limit = mean_bias + 1.96 * std_diff
    lower_limit = mean_bias - 1.96 * std_diff
    
    return mean_bias, lower_limit, upper_limit

def wilcoxon_signed_rank(y_true, y_pred):
    """
    Non-parametric test for paired samples.
    """
    stat, p_value = stats.wilcoxon(y_true, y_pred)
    return stat, p_value

def spearman_correlation(y_true, y_pred):
    coef, p = stats.spearmanr(y_true, y_pred)
    return coef, p

def pearson_correlation(y_true, y_pred):
    coef, p = stats.pearsonr(y_true, y_pred)
    return coef, p

def kruskal_wallis_dunns(df, value_col, group_col):
    """
    Performs Kruskal-Wallis H-test followed by Dunn's post-hoc test if significant.
    """
    groups = [group[value_col].values for name, group in df.groupby(group_col)]
    stat, p = stats.kruskal(*groups)
    
    results = {'kruskal_stat': stat, 'kruskal_p': p}
    
    if p < 0.05:
        import scikit_posthocs as sp
        dunn = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust='fdr_bh')
        results['dunn_posthoc'] = dunn
        
    return results

def friedman_test(data_matrix):
    """
    Friedman test for repeated measures (non-parametric alternative to repeated measures ANOVA).
    
    Args:
        data_matrix: 2D array where rows are samples (e.g., folds) and columns are treatments (models)
    
    Returns:
        stat: Test statistic
        p_value: p-value
    """
    stat, p_value = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(data_matrix.shape[1])])
    return stat, p_value

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.
    
    Interpretation:
    - Small effect: d ≈ 0.2
    - Medium effect: d ≈ 0.5
    - Large effect: d ≈ 0.8
    
    Args:
        group1, group2: Arrays of measurements
        
    Returns:
        d: Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence_level=0.95, statistic=np.mean):
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Array of measurements
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        statistic: Function to compute statistic (default: mean)
        
    Returns:
        lower, upper: Confidence interval bounds
        estimate: Point estimate
    """
    data = np.asarray(data)
    estimates = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        estimates.append(statistic(sample))
    
    estimates = np.array(estimates)
    alpha = 1 - confidence_level
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    estimate = statistic(data)
    
    return lower, upper, estimate

def compare_models_statistical(model_results, metric='val_kappa'):
    """
    Comprehensive statistical comparison of multiple models across folds.
    
    Args:
        model_results: Dict of {model_name: [scores_per_fold]}
        metric: Metric name for reporting
        
    Returns:
        dict with all statistical test results
    """
    model_names = list(model_results.keys())
    n_models = len(model_names)
    
    # Create data matrix (folds x models)
    fold_data = []
    max_folds = max(len(scores) for scores in model_results.values())
    
    for i in range(max_folds):
        fold_scores = [model_results[name][i] if i < len(model_results[name]) else np.nan 
                       for name in model_names]
        fold_data.append(fold_scores)
    
    data_matrix = np.array(fold_data)
    
    # Remove rows with any NaN (incomplete folds)
    data_matrix = data_matrix[~np.isnan(data_matrix).any(axis=1)]
    
    results = {
        'model_names': model_names,
        'metric': metric,
        'n_folds': len(data_matrix),
    }
    
    # Overall statistics per model
    results['model_stats'] = {}
    for i, name in enumerate(model_names):
        scores = data_matrix[:, i]
        lower, upper, mean = bootstrap_confidence_interval(scores)
        results['model_stats'][name] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores, ddof=1)),
            'median': float(np.median(scores)),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'scores': scores.tolist(),
        }
    
    # Friedman test (if more than 2 models)
    if n_models > 2:
        stat, p = friedman_test(data_matrix)
        results['friedman'] = {
            'statistic': float(stat),
            'p_value': float(p),
            'significant': p < 0.05,
        }
    
    # Pairwise comparisons (Wilcoxon + Effect Size)
    # Phase 1: collect raw p-values and pair keys for BH correction
    pair_keys = []
    raw_p_values = []
    pair_data = {}

    for i in range(n_models):
        for j in range(i + 1, n_models):
            name1, name2 = model_names[i], model_names[j]
            key = f'{name1}_vs_{name2}'
            scores1, scores2 = data_matrix[:, i], data_matrix[:, j]

            try:
                w_stat, w_p = stats.wilcoxon(scores1, scores2)
                cohen_d = cohens_d(scores1, scores2)
                pair_keys.append(key)
                raw_p_values.append(w_p)
                pair_data[key] = {
                    'wilcoxon_statistic': float(w_stat),
                    'wilcoxon_p_value': float(w_p),
                    'cohens_d': float(cohen_d),
                    'effect_size': 'large' if abs(cohen_d) >= 0.8 else 'medium' if abs(cohen_d) >= 0.5 else 'small' if abs(cohen_d) >= 0.2 else 'negligible',
                }
            except ValueError:
                pair_data[key] = {'note': 'Identical results', 'cohens_d': 0.0}

    # Phase 2: Benjamini-Hochberg correction on all raw p-values
    if raw_p_values:
        from statsmodels.stats.multitest import multipletests
        _, p_adjusted, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
        for key, p_adj in zip(pair_keys, p_adjusted):
            pair_data[key]['wilcoxon_p_value_adjusted_bh'] = float(p_adj)
            pair_data[key]['significant'] = p_adj < 0.05

    results['pairwise'] = pair_data

    return results
