"""Ranking engine to determine best models."""
from typing import Dict, Any

class RankingEngine:
    """Automatically ranks and recommends the best models."""
    
    @staticmethod
    def generate_recommendations(model_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Determine best models and generate textual recommendations."""
        if not model_metrics:
            return {}
            
        best_accuracy = max(model_metrics.items(), key=lambda x: x[1].get("accuracy", 0))
        best_latency = min(model_metrics.items(), key=lambda x: x[1].get("avg_latency", float('inf')))
        
        balanced_scores = {
            m: metrics.get("f1", 0) / max(1.0, metrics.get("avg_latency", 1.0) * 0.1)
            for m, metrics in model_metrics.items()
        }
        best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
        best_production = best_balanced 
        
        return {
            "best_accuracy_model": best_accuracy[0],
            "best_latency_model": best_latency[0],
            "best_balanced_model": best_balanced[0],
            "best_production_model": best_production[0],
            "recommendation": (
                f"For highest accuracy, use {best_accuracy[0]}. "
                f"For strict latency constraints, use {best_latency[0]}. "
                f"Overall recommended for production: {best_production[0]}."
            )
        }
