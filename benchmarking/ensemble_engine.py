"""Ensemble methods engine."""
from typing import List, Dict, Any
from collections import defaultdict

class EnsembleEngine:
    """Implements majority voting, weighted voting, and confidence averaging."""
    
    @staticmethod
    def ensemble_predictions(
        predictions: List[Dict[str, Any]], 
        method: str = "majority", 
        weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """Combine predictions from multiple models for the same samples."""
        
        grouped = defaultdict(list)
        for p in predictions:
            grouped[p["sample_id"]].append(p)
            
        ensemble_results = []
        for sample_id, preds in grouped.items():
            if method == "majority":
                final_pred = EnsembleEngine._majority_vote(preds)
            elif method == "weighted" and weights:
                final_pred = EnsembleEngine._weighted_vote(preds, weights)
            elif method == "confidence_averaging":
                final_pred = EnsembleEngine._confidence_average(preds)
            else:
                final_pred = EnsembleEngine._majority_vote(preds)
                
            ensemble_results.append({
                "sample_id": sample_id,
                "ensemble_method": method,
                "final_prediction": final_pred
            })
            
        return ensemble_results
        
    @staticmethod
    def _majority_vote(preds: List[Dict[str, Any]]) -> float:
        votes = [1 if p["prediction"] >= 0.5 else 0 for p in preds]
        return float(sum(votes) > len(votes) / 2)
        
    @staticmethod
    def _weighted_vote(preds: List[Dict[str, Any]], weights: Dict[str, float]) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for p in preds:
            w = weights.get(p["model_name"], 1.0)
            weighted_sum += p["prediction"] * w
            total_weight += w
        return weighted_sum / max(1e-9, total_weight)
        
    @staticmethod
    def _confidence_average(preds: List[Dict[str, Any]]) -> float:
        conf_sum = 0.0
        total_conf = 0.0
        for p in preds:
            conf = p.get("confidence", 1.0)
            conf_sum += p["prediction"] * conf
            total_conf += conf
        return conf_sum / max(1e-9, total_conf)
