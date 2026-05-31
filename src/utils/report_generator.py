"""
Comprehensive Report Generator for Model Comparison
Generates PDF, Markdown, and HTML reports with statistical analysis.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import yaml
matplotlib.use('Agg')  # Non-interactive backend

from src.utils.stat_tests import (
    compare_models_statistical,
    bootstrap_confidence_interval,
)


class ReportGenerator:
    """Generate comprehensive model comparison reports."""
    
    def __init__(self, run_dir: Path):
        """
        Initialize report generator.
        
        Args:
            run_dir: Path to run directory (experiments/{run_id})
        """
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.figures = []
        
        # Load all model results
        self.model_results = self._load_all_results()
        
        # Load baseline from config
        self.baseline = self._load_baseline()
        
    def _load_baseline(self) -> Optional[Dict]:
        """Load baseline metrics from configs/baseline.yaml"""
        baseline_path = Path(__file__).parent.parent.parent / "configs" / "baseline.yaml"
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                return yaml.safe_load(f)
        return None
        
    def _load_all_results(self) -> Dict:
        """Load results from all models in the run."""
        results = {}
        
        for model_dir in self.run_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            # Use figures/ instead of metrics/
            metrics_dir = model_dir / "figures"

            if not metrics_dir.exists():
                continue

            # Load metrics from all folds
            fold_metrics = []
            for metrics_file in sorted(metrics_dir.glob("fold_*_metrics.json")):
                with open(metrics_file, 'r') as f:
                    fold_metrics.append(json.load(f))

            # If no fold metrics, try final_metrics
            if not fold_metrics:
                final_metrics_path = metrics_dir / "final_metrics.json"
                if final_metrics_path.exists():
                    with open(final_metrics_path, 'r') as f:
                        fold_metrics = [json.load(f)]

            # Load model complexity
            complexity_path = metrics_dir / "model_complexity.json"
            complexity = None
            if complexity_path.exists():
                with open(complexity_path, 'r') as f:
                    complexity = json.load(f)

            # Load per-class metrics from all folds
            per_class_list = []
            for per_class_file in sorted(metrics_dir.glob("fold_*_per_class.json")):
                with open(per_class_file, 'r') as f:
                    per_class_list.append(json.load(f))

            # If no fold per-class, try generic
            if not per_class_list:
                per_class_path = metrics_dir / "per_class_metrics.json"
                if per_class_path.exists():
                    with open(per_class_path, 'r') as f:
                        per_class_list = [json.load(f)]

            results[model_name] = {
                'fold_metrics': fold_metrics,
                'complexity': complexity,
                'per_class': per_class_list,  # List of per-class dicts (one per fold)
            }
        
        return results
    
    def generate_all(self):
        """Generate all report formats."""
        print(f"\n{'='*80}")
        print(f"GENERATING REPORTS FOR RUN: {self.run_id}")
        print(f"{'='*80}\n")
        
        # Generate figures
        self._create_figures()
        
        # Generate reports
        pdf_path = self.run_dir / "report.pdf"
        md_path = self.run_dir / "report.md"
        html_path = self.run_dir / "report.html"
        
        self.generate_pdf(pdf_path)
        self.generate_markdown(md_path)
        self.generate_html(html_path)
        
        print(f"\nReports generated:")
        print(f"  - PDF:  {pdf_path}")
        print(f"  - MD:   {md_path}")
        print(f"  - HTML: {html_path}\n")
    
    def _create_figures(self):
        """Create all visualization figures."""
        # Create figures directory
        figures_dir = self.run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # 1. Metrics comparison bar chart
        self._create_metrics_comparison(figures_dir)
        
        # 2. Confusion matrices
        self._create_confusion_matrices(figures_dir)
        
        # 3. Per-class performance heatmap
        self._create_per_class_heatmap(figures_dir)
        
        # 4. Model complexity comparison
        self._create_complexity_comparison(figures_dir)
        
        # 5. Baseline comparison
        if self.baseline:
            self._create_baseline_comparison(figures_dir)
        
        # 6. QWK boxplot across folds per model
        self._create_qwk_boxplot(figures_dir)
        
        # 7. Per-class F1 boxplot across models
        self._create_per_class_f1_boxplot(figures_dir)
        
        plt.close('all')
    
    def _create_metrics_comparison(self, figures_dir: Path):
        """Create bar chart comparing all models."""
        metrics_to_plot = ['val_kappa', 'val_acc', 'val_macro_f1']
        metric_names = ['QWK', 'Accuracy', 'Macro-F1']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx]
            
            model_names = []
            means = []
            stds = []
            cis_lower = []
            cis_upper = []
            
            for model_name, data in self.model_results.items():
                scores = [m.get(metric, 0) for m in data['fold_metrics']]
                if scores:
                    model_names.append(model_name)
                    mean = np.mean(scores)
                    std = np.std(scores, ddof=1) if len(scores) > 1 else 0
                    
                    # Bootstrap CI
                    if len(scores) > 1:
                        ci_low, ci_up, _ = bootstrap_confidence_interval(scores)
                        cis_lower.append(mean - ci_low)
                        cis_upper.append(ci_up - mean)
                    else:
                        cis_lower.append(0)
                        cis_upper.append(0)
                    
                    means.append(mean)
                    stds.append(std)
            
            # Sort by mean descending
            sorted_indices = np.argsort(means)[::-1]
            model_names = [model_names[i] for i in sorted_indices]
            means = [means[i] for i in sorted_indices]
            cis_lower = [cis_lower[i] for i in sorted_indices]
            cis_upper = [cis_upper[i] for i in sorted_indices]
            
            # Plot
            x = np.arange(len(model_names))
            bars = ax.bar(x, means, yerr=[cis_lower, cis_upper], capsize=5, alpha=0.7)

            # Color best model differently, but check for empty bars
            if bars and len(bars) > 0:
                bars[0].set_color('green')
                bars[0].set_alpha(0.9)
            else:
                import logging
                logging.getLogger(__name__).error(
                    f"No bars to color in metrics comparison plot for '{name}'. Data may be empty."
                )

            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Comparison (with 95% CI)')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        path = figures_dir / "metrics_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        self.figures.append(('metrics_comparison', path))
        plt.close()
    
    def _create_confusion_matrices(self, figures_dir: Path):
        """Create confusion matrices for all models."""
        # Note: This requires confusion matrices to be saved during training
        # For now, create placeholder
        pass
    
    def _create_per_class_heatmap(self, figures_dir: Path):
        """Create heatmap of per-class F1 scores."""
        # Collect per-class metrics
        data = []
        model_names = []
        class_keys = [f'Class {i}' for i in range(4)]
        
        for model_name, results in self.model_results.items():
            if results['per_class']:
                model_names.append(model_name)
                # Average across folds
                per_class = results['per_class']
                if isinstance(per_class, list):
                    # Multiple folds
                    avg_f1 = {}
                    for fold_data in per_class:
                        for cls, metrics in fold_data.items():
                            if cls not in avg_f1:
                                avg_f1[cls] = []
                            avg_f1[cls].append(metrics.get('f1', 0))
                    data.append([np.mean(avg_f1.get(k, [0])) for k in class_keys])
                else:
                    data.append([per_class.get(k, {}).get('f1', 0) for k in class_keys])
        
        if data:
            fig, ax = plt.subplots(figsize=(8, len(model_names) * 0.5 + 2))
            sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                       yticklabels=model_names, ax=ax, vmin=0, vmax=1)
            ax.set_title('Per-Class F1 Scores')
            plt.tight_layout()
            path = figures_dir / "per_class_heatmap.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            self.figures.append(('per_class_heatmap', path))
            plt.close()
    
    def _create_complexity_comparison(self, figures_dir: Path):
        """Create model complexity comparison."""
        model_names = []
        params = []
        sizes = []
        times = []
        
        for model_name, data in self.model_results.items():
            if data['complexity']:
                model_names.append(model_name)
                params.append(data['complexity']['total_parameters'] / 1e6)  # Millions
                sizes.append(data['complexity']['model_size_mb'])
                times.append(data['complexity']['inference_time_ms_mean'])
        
        if model_names:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Parameters
            axes[0].barh(model_names, params)
            axes[0].set_xlabel('Parameters (Millions)')
            axes[0].set_title('Model Parameters')
            
            # Size
            axes[1].barh(model_names, sizes, color='orange')
            axes[1].set_xlabel('Size (MB)')
            axes[1].set_title('Model Size on Disk')
            
            # Inference time
            axes[2].barh(model_names, times, color='green')
            axes[2].set_xlabel('Inference Time (ms)')
            axes[2].set_title('Inference Time per Batch')
            
            plt.tight_layout()
            path = figures_dir / "complexity_comparison.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            self.figures.append(('complexity_comparison', path))
            plt.close()
    
    def _create_baseline_comparison(self, figures_dir: Path):
        """Create baseline comparison figure comparing current models with original paper."""
        if not self.baseline:
            return
        
        # Extract baseline accuracy from config
        baseline_acc = self.baseline['baseline']['performance']['accuracy']
        baseline_name = self.baseline['baseline']['model']['name']
        baseline_paper = f"{self.baseline['baseline']['paper']['authors']} ({self.baseline['baseline']['paper']['year']})"
        
        # Get current models' accuracies
        model_names = []
        accuracies = []
        qwks = []
        f1_scores = []
        
        for model_name, data in self.model_results.items():
            if data['fold_metrics']:
                model_names.append(model_name)
                
                # Calculate mean accuracy across folds
                acc_values = [fold.get('val_acc', 0) for fold in data['fold_metrics']]
                accuracies.append(np.mean(acc_values))
                
                # Calculate mean QWK
                qwk_values = [fold.get('val_kappa', 0) for fold in data['fold_metrics']]
                qwks.append(np.mean(qwk_values))
                
                # Calculate mean Macro F1
                f1_values = [fold.get('val_macro_f1', 0) for fold in data['fold_metrics']]
                f1_scores.append(np.mean(f1_values))
        
        if not model_names:
            return
        
        # Create comparison figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Accuracy comparison with baseline
        all_models = [f"{baseline_name}\n(Baseline)"] + model_names
        all_accuracies = [baseline_acc] + accuracies
        colors = ['#FF6B6B'] + ['#4ECDC4'] * len(model_names)
        
        bars = axes[0].barh(range(len(all_models)), all_accuracies, color=colors)
        axes[0].set_yticks(range(len(all_models)))
        axes[0].set_yticklabels(all_models)
        axes[0].set_xlabel('Accuracy', fontsize=12)
        axes[0].set_title(f'Accuracy Comparison\n(Baseline: {baseline_paper})', fontsize=14)
        axes[0].axvline(x=baseline_acc, color='red', linestyle='--', alpha=0.5, label=f'Baseline: {baseline_acc:.1%}')
        axes[0].set_xlim([0, 1.0])
        axes[0].legend()
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, all_accuracies)):
            width = bar.get_width()
            improvement = ""
            if i > 0:  # Not baseline
                diff = (acc - baseline_acc) / baseline_acc * 100
                improvement = f" ({diff:+.1f}%)"
            axes[0].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{acc:.3f}{improvement}', 
                        va='center', fontsize=9)
        
        # Plot 2: Multi-metric comparison (current models only)
        x = np.arange(len(model_names))
        width_bar = 0.25
        
        bars1 = axes[1].bar(x - width_bar, accuracies, width_bar, label='Accuracy', color='#4ECDC4')
        bars2 = axes[1].bar(x, qwks, width_bar, label='QWK', color='#95E1D3')
        bars3 = axes[1].bar(x + width_bar, f1_scores, width_bar, label='Macro F1', color='#F38181')
        
        axes[1].set_xlabel('Models', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Current Models: Multi-Metric Performance', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].axhline(y=baseline_acc, color='red', linestyle='--', alpha=0.5, label=f'Baseline Acc: {baseline_acc:.3f}')
        axes[1].legend()
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        path = figures_dir / "baseline_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.figures.append(("baseline_comparison", path))
    
    def _create_qwk_boxplot(self, figures_dir: Path):
        """Create boxplot of QWK distribution over folds per model.
        
        Shows stability and consistency of each model across CV folds.
        Particularly informative for random_stratified (5 folds).
        """
        model_names: List[str] = []
        qwk_per_model: List[List[float]] = []
        
        for model_name, data in self.model_results.items():
            scores = [m.get('val_kappa', 0) for m in data['fold_metrics']]
            if scores:
                model_names.append(model_name)
                qwk_per_model.append(scores)
        
        if not model_names:
            return
        
        # Sort by median QWK descending
        medians = [np.median(s) for s in qwk_per_model]
        sorted_idx = np.argsort(medians)[::-1]
        model_names = [model_names[i] for i in sorted_idx]
        qwk_per_model = [qwk_per_model[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.2), 6))
        
        bp = ax.boxplot(
            qwk_per_model,
            labels=model_names,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
            medianprops=dict(color='black', linewidth=1.5),
            widths=0.6,
        )
        
        # Color boxes with a gradient based on median rank
        cmap = plt.cm.RdYlGn
        n_models = len(model_names)
        for i, box in enumerate(bp['boxes']):
            color = cmap(1.0 - i / max(n_models - 1, 1))
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        # Overlay individual fold points for transparency
        for i, scores in enumerate(qwk_per_model):
            x_jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(scores))
            ax.scatter(
                [i + 1 + xj for xj in x_jitter], scores,
                color='black', alpha=0.6, s=30, zorder=5,
            )
        
        ax.set_ylabel('Quadratic Weighted Kappa (QWK)', fontsize=12)
        ax.set_title('QWK Distribution Over CV Folds Per Model', fontsize=14)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.05))
        
        # Legend for mean marker
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                   markersize=6, label='Mean'),
            Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=6, alpha=0.6, label='Fold values'),
        ]
        ax.legend(handles=legend_elements, loc='lower left')
        
        plt.tight_layout()
        path = figures_dir / "qwk_boxplot.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        self.figures.append(('qwk_boxplot', path))
        plt.close()
    
    def _create_per_class_f1_boxplot(self, figures_dir: Path):
        """Create boxplot of per-class F1 scores across all models.
        
        One box per class, showing the distribution of F1 values
        across all models (mean over folds first, then compare models).
        Reveals which classes are systemically difficult.
        """
        class_keys = [f'Class {i}' for i in range(4)]
        # Collect: for each model, the mean F1 per class across folds
        model_names: List[str] = []
        # per_class_f1[class_idx] = list of mean F1 values (one per model)
        per_class_f1: Dict[str, List[float]] = {k: [] for k in class_keys}
        
        for model_name, results in self.model_results.items():
            if not results['per_class']:
                continue
            
            per_class = results['per_class']
            if not isinstance(per_class, list):
                per_class = [per_class]
            
            # Average F1 across folds for each class
            class_f1_means: Dict[str, List[float]] = {}
            for fold_data in per_class:
                for cls, metrics in fold_data.items():
                    if cls not in class_f1_means:
                        class_f1_means[cls] = []
                    class_f1_means[cls].append(metrics.get('f1', 0))
            
            model_names.append(model_name)
            for cls_key in class_keys:
                mean_f1 = np.mean(class_f1_means.get(cls_key, [0]))
                per_class_f1[cls_key].append(mean_f1)
        
        if not model_names:
            return
        
        # Build data for boxplot: 4 boxes (one per class)
        box_data = [per_class_f1[k] for k in class_keys]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bp = ax.boxplot(
            box_data,
            labels=class_keys,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
            medianprops=dict(color='black', linewidth=1.5),
            widths=0.5,
        )
        
        # Color each class box distinctly
        class_colors = ['#4ECDC4', '#95E1D3', '#F38181', '#FF6B6B']
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(class_colors[i])
            box.set_alpha(0.7)
        
        # Overlay individual model points with labels
        for i, cls_key in enumerate(class_keys):
            scores = per_class_f1[cls_key]
            x_jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(scores))
            ax.scatter(
                [i + 1 + xj for xj in x_jitter], scores,
                color='black', alpha=0.6, s=30, zorder=5,
            )
            # Annotate outliers (lowest per class) with model name
            if scores:
                min_idx = int(np.argmin(scores))
                ax.annotate(
                    model_names[min_idx],
                    xy=(i + 1 + x_jitter[min_idx], scores[min_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    fontsize=7, alpha=0.8,
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                )
        
        ax.set_ylabel('F1-Score (mean over folds)', fontsize=12)
        ax.set_title('Per-Class F1 Distribution Across Models', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        # Add model count annotation
        ax.text(
            0.98, 0.02, f'n = {len(model_names)} models',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, style='italic', alpha=0.7,
        )
        
        plt.tight_layout()
        path = figures_dir / "per_class_f1_boxplot.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        self.figures.append(('per_class_f1_boxplot', path))
        plt.close()
    
    def _get_statistical_analysis(self) -> Dict:
        """Perform statistical analysis across models."""
        # Prepare data for statistical tests
        model_kappas = {}
        for model_name, data in self.model_results.items():
            scores = [m.get('val_kappa', 0) for m in data['fold_metrics']]
            if scores:
                model_kappas[model_name] = scores
        
        if len(model_kappas) < 2:
            return {"note": "Need at least 2 models for comparison"}
        
        return compare_models_statistical(model_kappas, metric='val_kappa')
    
    def generate_pdf(self, output_path: Path):
        """Generate PDF report."""
        with PdfPages(output_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, f'Model Comparison Report', 
                    ha='center', fontsize=24, weight='bold')
            fig.text(0.5, 0.6, f'Run ID: {self.run_id}', 
                    ha='center', fontsize=14)
            fig.text(0.5, 0.5, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                    ha='center', fontsize=12)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add all figures
            for fig_name, fig_path in self.figures:
                if fig_path.exists():
                    img = plt.imread(fig_path)
                    fig = plt.figure(figsize=(11, 8.5))
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
    
    def generate_markdown(self, output_path: Path):
        """Generate Markdown report."""
        lines = []
        lines.append(f"# Model Comparison Report")
        lines.append(f"\n**Run ID:** `{self.run_id}`")
        lines.append(f"\n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")
        
        # Summary table
        lines.append("## Summary Table\n")
        summary_data = []
        for model_name, data in self.model_results.items():
            scores_kappa = [m.get('val_kappa', 0) for m in data['fold_metrics']]
            scores_acc = [m.get('val_acc', 0) for m in data['fold_metrics']]
            scores_f1 = [m.get('val_macro_f1', 0) for m in data['fold_metrics']]
            
            if scores_kappa:
                summary_data.append({
                    'Model': model_name,
                    'QWK': f"{np.mean(scores_kappa):.4f} ± {np.std(scores_kappa, ddof=1):.4f}",
                    'Accuracy': f"{np.mean(scores_acc):.4f} ± {np.std(scores_acc, ddof=1):.4f}",
                    'Macro-F1': f"{np.mean(scores_f1):.4f} ± {np.std(scores_f1, ddof=1):.4f}",
                    'Folds': len(scores_kappa),
                })
        
        df = pd.DataFrame(summary_data)
        lines.append(df.to_markdown(index=False))
        lines.append("\n")
        
        # Baseline comparison section
        if self.baseline:
            lines.append("## Baseline Comparison\n")
            baseline_info = self.baseline['baseline']
            lines.append(f"**Original Paper:** {baseline_info['paper']['title']}\n")
            lines.append(f"**Authors:** {baseline_info['paper']['authors']}\n")
            lines.append(f"**Year:** {baseline_info['paper']['year']}\n")
            lines.append(f"**Journal:** {baseline_info['paper']['journal']}\n")
            lines.append(f"**DOI:** [{baseline_info['paper']['doi']}](https://doi.org/{baseline_info['paper']['doi']})\n\n")
            
            lines.append(f"**Baseline Model:** {baseline_info['model']['name']} ({baseline_info['model']['architecture']})\n")
            lines.append(f"**Input Size:** {baseline_info['model']['input_size']['resized']}\n")
            lines.append(f"**Training Data:** {baseline_info['training']['num_tiles']} manually labeled tiles\n")
            lines.append(f"**Baseline Accuracy:** {baseline_info['performance']['accuracy']:.1%}\n\n")
            
            lines.append("**Key Findings from Original Paper:**\n")
            lines.append(f"- Inter-annotator agreement (2 human experts): {baseline_info['performance']['human_agreement']:.1%}\n")
            lines.append(f"- CNN vs Ground Truth: {baseline_info['performance']['cnn_agreement']:.1%}\n")
            lines.append(f"- Most difficult class: Class 2 ({baseline_info['performance']['per_class']['class_2']:.1%} accuracy)\n")
            lines.append(f"- {baseline_info['performance']['notes'][0]}\n\n")
            
            lines.append("**Current Work Improvements:**\n")
            lines.append("- Multiple modern architectures explored (EfficientNet, ViT, Swin, etc.)\n")
            lines.append("- Additional metrics tracked: QWK (for ordinal data), Macro-F1, Per-class metrics\n")
            lines.append("- 5-fold cross-validation for robust performance estimation\n")
            lines.append("- Comprehensive statistical analysis with effect sizes\n\n")
        
        # Statistical analysis
        lines.append("## Statistical Analysis\n")
        stats = self._get_statistical_analysis()
        if 'friedman' in stats:
            lines.append(f"**Friedman Test:** χ² = {stats['friedman']['statistic']:.4f}, "
                        f"p = {stats['friedman']['p_value']:.4f} "
                        f"({'significant' if stats['friedman']['significant'] else 'not significant'})\n")
        
        # Pairwise comparisons
        if 'pairwise' in stats:
            lines.append("\n### Pairwise Comparisons\n")
            for pair, results in stats['pairwise'].items():
                if 'wilcoxon_p_value' in results:
                    lines.append(f"- **{pair}**: p = {results['wilcoxon_p_value']:.4f}, "
                                f"Cohen's d = {results['cohens_d']:.3f} ({results['effect_size']} effect)\n")
        
        # Figures
        lines.append("\n## Visualizations\n")
        for fig_name, fig_path in self.figures:
            rel_path = fig_path.relative_to(self.run_dir)
            lines.append(f"\n### {fig_name.replace('_', ' ').title()}\n")
            lines.append(f"![{fig_name}]({rel_path})\n")
        
        # Winner
        lines.append("\n## Recommendation\n")
        if summary_data:
            # Find best by QWK
            best_idx = np.argmax([float(d['QWK'].split('±')[0]) for d in summary_data])
            best_model = summary_data[best_idx]['Model']
            lines.append(f"**Winner:** `{best_model}`\n")
            lines.append(f"- Best Quadratic Weighted Kappa: {summary_data[best_idx]['QWK']}\n")
        
        output_path.write_text('\n'.join(lines))
    
    def generate_html(self, output_path: Path):
        """Generate HTML report."""
        md_path = output_path.with_suffix('.md')
        if md_path.exists():
            # Convert markdown to HTML (simple version)
            import markdown
            md_content = md_path.read_text()
            html_content = markdown.markdown(md_content, extensions=['tables'])
            
            # Wrap in HTML template
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report - {self.run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        img {{ max-width: 100%; height: auto; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
            output_path.write_text(html)
        else:
            print("Warning: Markdown file not found, HTML generation skipped")


# === Cross-Run Comparison Functions ===

def compare_multiple_runs(
    experiments_dir: str,
    run_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    include_statistical_tests: bool = True
) -> pd.DataFrame:
    """
    Compare models across multiple training runs (different run_ids).
    
    **Use Case:** You trained DenseNet on Monday, GNN on Tuesday, ViT on Wednesday.
    This function lets you compare all three models statistically.
    
    Args:
        experiments_dir: Path to experiments directory (e.g., "experiments/")
        run_ids: List of run IDs to compare. If None, compares ALL runs.
        output_dir: Where to save comparison report. If None, uses experiments_dir/cross_run_comparison/
        include_statistical_tests: Whether to run Friedman + Wilcoxon tests
        
    Returns:
        DataFrame with columns: Run ID, Model, Mean QWK, Std QWK, Mean Acc, etc.
        
    Example:
        >>> comparison_df = compare_multiple_runs(
        ...     experiments_dir="experiments/",
        ...     run_ids=["2026-01-07_10-30-00", "2026-01-08_09-15-00"],
        ...     include_statistical_tests=True
        ... )
        >>> print(comparison_df[['Model', 'Mean QWK', 'Mean Acc']])
    """
    experiments_path = Path(experiments_dir)
    
    # Auto-discover run_ids if not provided
    if run_ids is None:
        run_ids = [d.name for d in experiments_path.iterdir() if d.is_dir()]
        print(f"Auto-discovered {len(run_ids)} runs: {run_ids}")
    
    # Collect all model results across runs
    all_results = []
    
    for run_id in run_ids:
        run_dir = experiments_path / run_id
        if not run_dir.exists():
            print(f"Warning: Run {run_id} not found, skipping")
            continue
        
        # Iterate through all model directories in this run
        for model_dir in run_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            # Load metrics from all folds
            figures_dir = model_dir / "figures"
            if not figures_dir.exists():
                continue
            
            fold_kappas = []
            fold_accs = []

            # Collect per-fold metrics. Prefer canonical fold metrics files, but
            # also support legacy per-class naming for backward compatibility.
            metrics_files = sorted(figures_dir.glob("fold_*_metrics.json"))
            if not metrics_files:
                metrics_files = sorted(figures_dir.glob("fold_*_per_class.json"))
            if not metrics_files:
                metrics_files = sorted(figures_dir.glob("per_class_metrics_fold*.json"))

            for metrics_file in metrics_files:
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)

                    # Canonical metrics files usually store val_kappa/val_acc,
                    # while per-class files can store quadratic_weighted_kappa/accuracy.
                    qwk = metrics.get('val_kappa', metrics.get('quadratic_weighted_kappa', np.nan))
                    acc = metrics.get('val_acc', metrics.get('accuracy', np.nan))

                    fold_kappas.append(qwk)
                    fold_accs.append(acc)
                except Exception as e:
                    print(f"Warning: Failed to load {metrics_file}: {e}")
                    continue
            
            # Skip if no valid metrics found
            if not fold_kappas:
                continue
            
            # Calculate statistics
            all_results.append({
                'Run ID': run_id,
                'Model': model_name,
                'Mean QWK': np.mean(fold_kappas),
                'Std QWK': np.std(fold_kappas, ddof=1) if len(fold_kappas) > 1 else 0.0,
                'Mean Acc': np.mean(fold_accs),
                'Std Acc': np.std(fold_accs, ddof=1) if len(fold_accs) > 1 else 0.0,
                'N Folds': len(fold_kappas),
                'Fold QWKs': fold_kappas,  # For statistical tests
                'Fold Accs': fold_accs
            })
    
    if not all_results:
        raise ValueError("No valid results found in any run!")
    
    # Create DataFrame
    comparison_df = pd.DataFrame(all_results)
    
    # Sort by Mean QWK (descending)
    comparison_df = comparison_df.sort_values('Mean QWK', ascending=False)
    
    # Setup output directory
    if output_dir is None:
        output_dir = experiments_path / "cross_run_comparison"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save comparison table
    table_path = output_path / "comparison_table.csv"
    comparison_df[['Run ID', 'Model', 'Mean QWK', 'Std QWK', 'Mean Acc', 'Std Acc', 'N Folds']].to_csv(
        table_path, index=False
    )
    print(f"Comparison table saved to {table_path}")
    
    # Generate statistical tests if requested
    if include_statistical_tests and len(comparison_df) >= 2:
        _generate_cross_run_statistical_tests(comparison_df, output_path)
    
    # Generate comparison plots
    _generate_cross_run_plots(comparison_df, output_path)
    
    return comparison_df


def _generate_cross_run_statistical_tests(df: pd.DataFrame, output_dir: Path):
    """Generate statistical tests for cross-run comparison."""
    from src.utils.stat_tests import compare_models_statistical
    
    # Prepare data for statistical tests (need arrays of fold scores per model)
    model_scores = {}
    for _, row in df.iterrows():
        model_key = f"{row['Run ID']}_{row['Model']}"
        model_scores[model_key] = row['Fold QWKs']
    
    # Run statistical comparison
    try:
        stat_results = compare_models_statistical(
            model_results=model_scores,
            metric='val_kappa'
        )
        
        # Save results
        stat_path = output_dir / "statistical_tests.json"
        with open(stat_path, 'w') as f:
            json.dump(stat_results, f, indent=2)
        
        print(f"Statistical tests saved to {stat_path}")
    except Exception as e:
        print(f"Warning: Statistical tests failed: {e}")


def _generate_cross_run_plots(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots for cross-run analysis."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Bar plot: Mean QWK comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create combined labels (Run ID + Model)
    df['Label'] = df['Run ID'].str[-8:] + '\n' + df['Model']  # Last 8 chars of run_id + model
    
    x = np.arange(len(df))
    bars = ax.bar(x, df['Mean QWK'], yerr=df['Std QWK'], capsize=5, alpha=0.7)
    
    # Color bars by run_id
    unique_runs = df['Run ID'].unique()
    colors = sns.color_palette("husl", len(unique_runs))
    run_colors = {run: color for run, color in zip(unique_runs, colors)}
    
    for bar, run_id in zip(bars, df['Run ID']):
        bar.set_color(run_colors[run_id])
    
    ax.set_xlabel('Model (Run ID)', fontsize=12)
    ax.set_ylabel('Mean Quadratic Weighted Kappa', fontsize=12)
    ax.set_title('Cross-Run Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Label'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend for runs
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=run_colors[run], label=run) for run in unique_runs]
    ax.legend(handles=legend_elements, title='Run ID', loc='upper right')
    
    plt.tight_layout()
    plot_path = output_dir / "cross_run_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {plot_path}")
    
    # 2. Scatter plot: QWK vs Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for run_id in unique_runs:
        run_data = df[df['Run ID'] == run_id]
        ax.scatter(
            run_data['Mean Acc'], 
            run_data['Mean QWK'],
            s=100,
            alpha=0.7,
            label=run_id,
            color=run_colors[run_id]
        )
        
        # Add model labels
        for _, row in run_data.iterrows():
            ax.annotate(
                row['Model'],
                (row['Mean Acc'], row['Mean QWK']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
    
    ax.set_xlabel('Mean Accuracy', fontsize=12)
    ax.set_ylabel('Mean Quadratic Weighted Kappa', fontsize=12)
    ax.set_title('Cross-Run: QWK vs Accuracy', fontsize=14, fontweight='bold')
    ax.legend(title='Run ID', loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    scatter_path = output_dir / "cross_run_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plot saved to {scatter_path}")
