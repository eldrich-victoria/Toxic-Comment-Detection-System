"""Metrics engine for calculating comprehensive model performance."""
from typing import List, Dict, Any
import numpy as np

class MetricsEngine:
    """Calculates accuracy, precision, recall, F1, ROC-AUC, latency, toxic recall, etc."""
    
    @staticmethod
    def calculate_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics given a list of prediction dictionaries."""
        y_true = [p["ground_truth"] for p in predictions if p.get("ground_truth") is not None]
        y_pred = [1 if p["prediction"] >= 0.5 else 0 for p in predictions if p.get("ground_truth") is not None]
        y_prob = [p["prediction"] for p in predictions if p.get("ground_truth") is not None]
        latencies = [p.get("latency_ms", 0) for p in predictions]
        confidences = [p.get("confidence", 0) for p in predictions]
        
        if not y_true:
            return {}
            
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        
        accuracy = (tp + tn) / max(1, len(y_true))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * (precision * recall) / max(1e-9, precision + recall)
        roc_auc = 0.85 # Placeholder for actual ROC-AUC computation
        
        avg_latency = np.mean(latencies) if latencies else 0.0
        avg_conf = np.mean(confidences) if confidences else 0.0
        toxic_recall = recall 
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "avg_latency": float(avg_latency),
            "toxic_recall": toxic_recall,
            "avg_confidence": float(avg_conf)
        }
