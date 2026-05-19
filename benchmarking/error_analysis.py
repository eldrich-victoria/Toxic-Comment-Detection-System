"""Error analysis and XAI integration."""
from typing import List, Dict, Any

class ErrorAnalyzer:
    """Tracks false positives, negatives, uncertainties, sarcasm, and adversarial failures."""
    
    @staticmethod
    def run_lime_explainability(text: str, prediction: float) -> Dict[str, float]:
        """Simulate LIME explainability integration."""
        return {"toxic_word_1": 0.45, "toxic_word_2": 0.30, "neutral_word": -0.10}
        
    @staticmethod
    def analyze_errors(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Categorize failures and generate XAI."""
        errors = []
        for p in predictions:
            y_true = p.get("ground_truth")
            if y_true is None:
                continue
                
            y_pred = 1 if p["prediction"] >= 0.5 else 0
            error_type = None
            
            if y_pred == 1 and y_true == 0:
                error_type = "False Positive"
            elif y_pred == 0 and y_true == 1:
                error_type = "False Negative"
            elif 0.4 < p["prediction"] < 0.6:
                error_type = "Uncertain Prediction"
                
            text = p.get("text_input", "").lower()
            if "sarcasm" in text and y_pred != y_true:
                error_type = "Sarcasm Failure"
            if "adversarial" in text and y_pred != y_true:
                error_type = "Adversarial Failure"
                
            if error_type:
                # Integrate LIME
                xai_features = ErrorAnalyzer.run_lime_explainability(text, p["prediction"])
                
                errors.append({
                    "prediction_id": p.get("id"),
                    "sample_id": p.get("sample_id"),
                    "error_type": error_type,
                    "analysis_notes": f"Model failed due to {error_type}",
                    "lime_explanation": xai_features
                })
                
        return errors
        
    @staticmethod
    def generate_confusion_matrix(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Generate a basic confusion matrix."""
        y_true = [p["ground_truth"] for p in predictions if p.get("ground_truth") is not None]
        y_pred = [1 if p["prediction"] >= 0.5 else 0 for p in predictions if p.get("ground_truth") is not None]
        
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
