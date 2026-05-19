"""Report generation system for exporting benchmark summaries."""
import json
from typing import Dict, Any, List
from pathlib import Path

from app.core.path_manager import PathManager
from app.core.logger import logger
from benchmarking.visualization_engine import VisualizationEngine

class ReportGenerator:
    """Generates PDF, DOCX, and HTML reports for benchmarks."""
    
    def __init__(self, run_id: str, run_name: str):
        self.run_id = str(run_id)
        self.run_name = run_name
        self.reports_dir = PathManager.BENCHMARK_RESULTS_DIR / "reports" / self.run_id
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = VisualizationEngine(self.run_id)

    def compile_report_data(
        self,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        rankings: Dict[str, Any],
        fairness: Dict[str, Any],
        drift: Dict[str, Any],
        errors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate all analytics data for reporting."""
        return {
            "executive_summary": "Benchmark run completed successfully. Review visualizations and metrics below.",
            "benchmark_metadata": metadata,
            "model_comparison": metrics,
            "rankings": rankings,
            "fairness_analysis": fairness,
            "drift_analysis": drift,
            "ensemble_analysis": {}, # Populated externally if needed
            "error_analysis": {"total_errors": len(errors), "sample_errors": errors[:5]},
            "explainability_outputs": [e.get("lime_explanation") for e in errors[:3] if "lime_explanation" in e],
            "production_recommendations": rankings.get("recommendation", "No recommendation available.")
        }

    def generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate an HTML interactive report using basic strings (Jinja2 for production)."""
        out_path = self.reports_dir / f"{self.run_name}_report.html"
        
        html_content = f"<html><head><title>Benchmark Report: {self.run_name}</title></head>"
        html_content += f"<body><h1>Executive Summary</h1><p>{data['executive_summary']}</p>"
        html_content += f"<h2>Production Recommendations</h2><p>{data['production_recommendations']}</p>"
        html_content += f"<h2>Model Metrics</h2><pre>{json.dumps(data['model_comparison'], indent=2)}</pre>"
        html_content += f"<h2>Error Analysis</h2><p>Total Errors: {data['error_analysis']['total_errors']}</p>"
        html_content += "</body></html>"
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"HTML Report generated at {out_path}")
        return str(out_path)

    def generate_pdf_report(self, data: Dict[str, Any]) -> str:
        """Generate a PDF report. (Mock implementation for architecture)"""
        out_path = self.reports_dir / f"{self.run_name}_report.pdf"
        
        # In a real production scenario, use fpdf2 or WeasyPrint here.
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"PDF Binary Placeholder.\nRun: {self.run_name}\nRecommendations: {data['production_recommendations']}")
            
        logger.info(f"PDF Report generated at {out_path}")
        return str(out_path)

    def generate_docx_report(self, data: Dict[str, Any]) -> str:
        """Generate a DOCX report. (Mock implementation for architecture)"""
        out_path = self.reports_dir / f"{self.run_name}_report.docx"
        
        # In a real production scenario, use python-docx here.
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"DOCX Binary Placeholder.\nRun: {self.run_name}\nRecommendations: {data['production_recommendations']}")
            
        logger.info(f"DOCX Report generated at {out_path}")
        return str(out_path)

    def generate_all_reports(self, data: Dict[str, Any]):
        """Generate all configured report formats and trigger visualizations."""
        # Trigger basic charts
        if "model_comparison" in data:
            self.visualizer.generate_ranking_chart(data["model_comparison"])
        if "fairness_analysis" in data:
            self.visualizer.generate_fairness_chart(data["fairness_analysis"])
        if "drift_analysis" in data:
            self.visualizer.generate_drift_chart(data["drift_analysis"])
            
        self.generate_html_report(data)
        self.generate_pdf_report(data)
        self.generate_docx_report(data)
