"""Visualization engine for generating charts and plots."""
import os
from typing import List, Dict, Any

# Delaying heavy imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import matplotlib.pyplot as plt
    HAS_VIS_LIBS = True
except ImportError:
    HAS_VIS_LIBS = False

from app.core.path_manager import PathManager
from app.core.logger import logger

class VisualizationEngine:
    """Generates and exports various benchmark charts."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.charts_dir = PathManager.BENCHMARK_RESULTS_DIR / "charts" / run_id
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        if not HAS_VIS_LIBS:
            logger.warning("Visualization libraries not installed. Charts will be skipped.")
        
    def generate_confusion_matrix(self, conf_matrix: Dict[str, int], model_name: str) -> str:
        """Generate and save confusion matrix plot."""
        if not HAS_VIS_LIBS: return ""
        
        fig, ax = plt.subplots(figsize=(6, 4))
        matrix = [[conf_matrix.get("TP", 0), conf_matrix.get("FP", 0)],
                  [conf_matrix.get("FN", 0), conf_matrix.get("TN", 0)]]
        
        cax = ax.matshow(matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), va='center', ha='center')
                
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        out_path = self.charts_dir / f"conf_matrix_{model_name}.png"
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved confusion matrix for {model_name}")
        return str(out_path)

    def generate_latency_histogram(self, latencies: List[float], model_name: str) -> str:
        """Generate interactive latency histogram using Plotly."""
        if not HAS_VIS_LIBS: return ""
        
        fig = px.histogram(x=latencies, nbins=20, title=f"Latency Distribution: {model_name}",
                           labels={'x': 'Latency (ms)', 'y': 'Count'})
        
        out_path = self.charts_dir / f"latency_{model_name}.html"
        fig.write_html(str(out_path))
        logger.info(f"Saved latency histogram for {model_name}")
        return str(out_path)

    def generate_ranking_chart(self, rankings: Dict[str, Any]) -> str:
        """Generate chart for model rankings."""
        if not HAS_VIS_LIBS: return ""
        
        models = list(rankings.keys())
        scores = [v.get("f1", 0) for v in rankings.values() if isinstance(v, dict)]
        if not scores:
            return ""
            
        fig = px.bar(x=models, y=scores, title="Model F1 Score Rankings",
                     labels={'x': 'Models', 'y': 'F1 Score'}, color=scores)
        
        out_path = self.charts_dir / "model_rankings.html"
        fig.write_html(str(out_path))
        return str(out_path)

    def generate_fairness_chart(self, fairness_results: Dict[str, Any]) -> str:
        """Generate fairness impact charts."""
        if not HAS_VIS_LIBS: return ""
        
        groups = list(fairness_results.keys())
        impacts = [v.get("disparate_impact_ratio", 1.0) for v in fairness_results.values() if isinstance(v, dict)]
        if not impacts:
            return ""
            
        fig = px.bar(x=groups, y=impacts, title="Fairness: Disparate Impact Ratio",
                     labels={'x': 'Bias Type', 'y': 'Impact Ratio (Ideal ~1.0)'})
        fig.add_hline(y=1.0, line_dash="dash", line_color="green")
        fig.add_hline(y=0.8, line_dash="dash", line_color="red")
        
        out_path = self.charts_dir / "fairness_analysis.html"
        fig.write_html(str(out_path))
        return str(out_path)
        
    def generate_drift_chart(self, drift_results: Dict[str, Any]) -> str:
        """Generate drift impact charts."""
        if not HAS_VIS_LIBS: return ""
        
        features = list(drift_results.keys())
        scores = [v.get("drift_score", 0.0) for v in drift_results.values() if isinstance(v, dict)]
        
        fig = px.line(x=features, y=scores, title="Data Drift Analysis", markers=True,
                     labels={'x': 'Feature', 'y': 'Drift Score (Higher = More Drift)'})
        
        out_path = self.charts_dir / "drift_analysis.html"
        fig.write_html(str(out_path))
        return str(out_path)
