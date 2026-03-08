import json
import os
import glob
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class CategoryStats:
    name: str
    total: int
    positive: int
    negative: int

    @property
    def balance_percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.positive / self.total) * 100

class DatasetLoader:
    """Handles loading JSON datasets from a directory."""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def get_json_files(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.data_dir, "*.json")))

    def load_category_data(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r') as f:
            return json.load(f)

class DataAnalyzer:
    """Analyzes the statistical distribution of sensitivity scores."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def analyze(self, category_name: str, data: List[Dict]) -> CategoryStats:
        pos = sum(1 for d in data if d.get('sensitivity_score', d.get('label', 0)) >= self.threshold)
        total = len(data)
        return CategoryStats(
            name=category_name,
            total=total,
            positive=pos,
            negative=total - pos
        )

class BalanceReporter:
    """Formats and prints the analysis results to the console."""
    @staticmethod
    def print_report(stats_list: List[CategoryStats]):
        header = f"{'Category':<20} | {'Total':<8} | {'Positive (>=0.5)':<18} | {'Negative (<0.5)':<18} | {'Balance %':<10}"
        separator = "-" * len(header)
        
        print(header)
        print(separator)
        
        for stats in stats_list:
            print(f"{stats.name:<20} | {stats.total:<8} | {stats.positive:<18} | {stats.negative:<18} | {stats.balance_percentage:>8.1f}%")

class DataBalanceChecker:
    """Orchestrator for checking data balance."""
    def __init__(self, data_dir: str):
        self.loader = DatasetLoader(data_dir)
        self.analyzer = DataAnalyzer()
        self.reporter = BalanceReporter()

    def run(self):
        files = self.loader.get_json_files()
        if not files:
            print(f"No JSON files found in {self.loader.data_dir}")
            return

        results = []
        for file in files:
            category_name = os.path.splitext(os.path.basename(file))[0]
            data = self.loader.load_category_data(file)
            stats = self.analyzer.analyze(category_name, data)
            results.append(stats)
        
        self.reporter.print_report(results)

if __name__ == "__main__":
    project_root = os.getcwd()
    data_path = os.path.join(project_root, "data")
    
    checker = DataBalanceChecker(data_path)
    checker.run()
