"""Evaluation metrics and leaderboard for workforce planning models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    total_cost: float
    service_level: float
    utilization_rate: float
    skill_coverage: float
    hiring_efficiency: float
    retention_rate: float
    overtime_hours: float
    skill_gaps: float
    cost_per_employee: float
    demand_satisfaction: float


class WorkforceEvaluator:
    """Evaluator for workforce planning models."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics_history = []
        self.baseline_metrics = None
        
    def calculate_service_level(
        self, 
        actual_staffing: Dict[str, int], 
        demand: Dict[str, int]
    ) -> float:
        """Calculate service level (demand satisfaction rate).
        
        Args:
            actual_staffing: Actual staffing levels
            demand: Demand requirements
            
        Returns:
            Service level as percentage
        """
        total_demand = sum(demand.values())
        total_satisfied = sum(min(actual_staffing.get(k, 0), v) for k, v in demand.items())
        
        return (total_satisfied / total_demand) if total_demand > 0 else 0.0
    
    def calculate_utilization_rate(
        self, 
        employees: pd.DataFrame, 
        demand_forecast: pd.DataFrame
    ) -> float:
        """Calculate workforce utilization rate.
        
        Args:
            employees: Employee data
            demand_forecast: Demand forecast
            
        Returns:
            Utilization rate as percentage
        """
        total_capacity = len(employees) * 40 * 4  # 40 hours/week * 4 weeks/month
        total_demand = demand_forecast['demand'].sum() * 40 * 4
        
        return min(1.0, total_demand / total_capacity) if total_capacity > 0 else 0.0
    
    def calculate_skill_coverage(
        self, 
        employees: pd.DataFrame, 
        skill_requirements: pd.DataFrame
    ) -> float:
        """Calculate skill coverage across departments.
        
        Args:
            employees: Employee data with skills
            skill_requirements: Required skills by department
            
        Returns:
            Skill coverage percentage
        """
        total_coverage = 0
        total_requirements = 0
        
        for _, req in skill_requirements.iterrows():
            dept = req['department']
            skill = req['skill']
            required_level = req['required_level']
            
            # Get employees in this department
            dept_employees = employees[employees['department'] == dept]
            
            if not dept_employees.empty:
                # Calculate average skill level
                skill_levels = []
                for _, emp in dept_employees.iterrows():
                    if skill in emp['skill_proficiencies']:
                        skill_levels.append(emp['skill_proficiencies'][skill])
                
                avg_skill_level = np.mean(skill_levels) if skill_levels else 0
                coverage = min(1.0, avg_skill_level / required_level) if required_level > 0 else 1.0
            else:
                coverage = 0
            
            total_coverage += coverage
            total_requirements += 1
        
        return (total_coverage / total_requirements) if total_requirements > 0 else 0.0
    
    def calculate_hiring_efficiency(
        self, 
        hires: Dict[str, int], 
        demand_gap: Dict[str, int]
    ) -> float:
        """Calculate hiring efficiency (hires vs demand gap).
        
        Args:
            hires: Number of hires
            demand_gap: Gap between demand and current staffing
            
        Returns:
            Hiring efficiency percentage
        """
        total_hires = sum(hires.values())
        total_gap = sum(demand_gap.values())
        
        return (total_gap / total_hires) if total_hires > 0 else 0.0
    
    def calculate_retention_rate(self, employees: pd.DataFrame) -> float:
        """Calculate average retention rate.
        
        Args:
            employees: Employee data with retention probabilities
            
        Returns:
            Average retention rate
        """
        return employees['retention_probability'].mean()
    
    def calculate_overtime_hours(self, overtime: Dict[str, float]) -> float:
        """Calculate total overtime hours.
        
        Args:
            overtime: Overtime hours by department/quarter
            
        Returns:
            Total overtime hours
        """
        return sum(overtime.values())
    
    def calculate_skill_gaps(
        self, 
        employees: pd.DataFrame, 
        skill_requirements: pd.DataFrame
    ) -> float:
        """Calculate skill gaps across organization.
        
        Args:
            employees: Employee data with skills
            skill_requirements: Required skills by department
            
        Returns:
            Average skill gap percentage
        """
        gaps = []
        
        for _, req in skill_requirements.iterrows():
            dept = req['department']
            skill = req['skill']
            required_level = req['required_level']
            
            dept_employees = employees[employees['department'] == dept]
            
            if not dept_employees.empty:
                skill_levels = []
                for _, emp in dept_employees.iterrows():
                    if skill in emp['skill_proficiencies']:
                        skill_levels.append(emp['skill_proficiencies'][skill])
                
                avg_skill_level = np.mean(skill_levels) if skill_levels else 0
                gap = max(0, required_level - avg_skill_level)
            else:
                gap = required_level
            
            gaps.append(gap)
        
        return np.mean(gaps)
    
    def calculate_cost_per_employee(
        self, 
        total_cost: float, 
        total_employees: int
    ) -> float:
        """Calculate cost per employee.
        
        Args:
            total_cost: Total workforce cost
            total_employees: Total number of employees
            
        Returns:
            Cost per employee
        """
        return total_cost / total_employees if total_employees > 0 else 0.0
    
    def evaluate_model(
        self,
        optimization_result: Dict[str, Any],
        demand_forecast: pd.DataFrame,
        employees: pd.DataFrame,
        skill_requirements: pd.DataFrame,
        model_name: str = "model"
    ) -> EvaluationMetrics:
        """Evaluate a workforce planning model.
        
        Args:
            optimization_result: Results from optimization
            demand_forecast: Demand forecast data
            employees: Employee data
            skill_requirements: Skill requirements
            model_name: Name of the model being evaluated
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Extract basic metrics
        total_cost = optimization_result.get('total_cost', 0)
        
        # Calculate service level
        if 'hires' in optimization_result:
            hires = optimization_result['hires']
            if isinstance(hires, dict):
                # Advanced model with department-wise hires
                actual_staffing = {}
                demand_dict = {}
                
                for _, row in demand_forecast.iterrows():
                    quarter = row['quarter']
                    dept = row['department']
                    demand_val = row['demand']
                    
                    key = f"{quarter}_{dept}"
                    actual_staffing[key] = hires.get((quarter, dept), 0)
                    demand_dict[key] = demand_val
                
                service_level = self.calculate_service_level(actual_staffing, demand_dict)
            else:
                # Basic model with quarterly hires
                total_hires = sum(hires)
                total_demand = demand_forecast['demand'].sum()
                service_level = min(1.0, total_hires / total_demand) if total_demand > 0 else 0.0
        else:
            service_level = 0.0
        
        # Calculate other metrics
        utilization_rate = self.calculate_utilization_rate(employees, demand_forecast)
        skill_coverage = self.calculate_skill_coverage(employees, skill_requirements)
        retention_rate = self.calculate_retention_rate(employees)
        
        # Calculate overtime if available
        overtime_hours = 0
        if 'overtime' in optimization_result:
            overtime_hours = self.calculate_overtime_hours(optimization_result['overtime'])
        
        # Calculate skill gaps
        skill_gaps = self.calculate_skill_gaps(employees, skill_requirements)
        
        # Calculate hiring efficiency
        hiring_efficiency = 0
        if 'hires' in optimization_result:
            hires = optimization_result['hires']
            if isinstance(hires, dict):
                total_hires = sum(hires.values())
                total_demand = demand_forecast['demand'].sum()
                hiring_efficiency = (total_demand / total_hires) if total_hires > 0 else 0.0
        
        # Calculate cost per employee
        total_employees = len(employees)
        cost_per_employee = self.calculate_cost_per_employee(total_cost, total_employees)
        
        # Calculate demand satisfaction
        demand_satisfaction = service_level  # Same as service level for now
        
        metrics = EvaluationMetrics(
            total_cost=total_cost,
            service_level=service_level,
            utilization_rate=utilization_rate,
            skill_coverage=skill_coverage,
            hiring_efficiency=hiring_efficiency,
            retention_rate=retention_rate,
            overtime_hours=overtime_hours,
            skill_gaps=skill_gaps,
            cost_per_employee=cost_per_employee,
            demand_satisfaction=demand_satisfaction
        )
        
        # Store metrics
        self.metrics_history.append({
            'model': model_name,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now()
        })
        
        logger.info(f"Evaluation completed for {model_name}")
        return metrics
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard of model performance.
        
        Returns:
            DataFrame with model rankings
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        leaderboard_data = []
        
        for entry in self.metrics_history:
            metrics = entry['metrics']
            leaderboard_data.append({
                'Model': entry['model'],
                'Total Cost': metrics.total_cost,
                'Service Level': metrics.service_level,
                'Utilization Rate': metrics.utilization_rate,
                'Skill Coverage': metrics.skill_coverage,
                'Hiring Efficiency': metrics.hiring_efficiency,
                'Retention Rate': metrics.retention_rate,
                'Overtime Hours': metrics.overtime_hours,
                'Skill Gaps': metrics.skill_gaps,
                'Cost per Employee': metrics.cost_per_employee,
                'Demand Satisfaction': metrics.demand_satisfaction,
                'Timestamp': entry['timestamp']
            })
        
        df = pd.DataFrame(leaderboard_data)
        
        # Calculate composite score (weighted average)
        weights = {
            'Service Level': 0.25,
            'Utilization Rate': 0.20,
            'Skill Coverage': 0.15,
            'Hiring Efficiency': 0.15,
            'Retention Rate': 0.10,
            'Demand Satisfaction': 0.15
        }
        
        # Normalize metrics (higher is better)
        for col in weights.keys():
            if col in df.columns:
                max_val = df[col].max()
                if max_val > 0:
                    df[f'{col}_normalized'] = df[col] / max_val
                else:
                    df[f'{col}_normalized'] = 0
        
        # Calculate composite score
        composite_score = 0
        for col, weight in weights.items():
            normalized_col = f'{col}_normalized'
            if normalized_col in df.columns:
                composite_score += df[normalized_col] * weight
        
        df['Composite Score'] = composite_score
        
        # Rank models
        df = df.sort_values('Composite Score', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        return df
    
    def plot_metrics_comparison(self, save_path: Optional[str] = None) -> None:
        """Plot comparison of metrics across models.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.metrics_history:
            logger.warning("No metrics to plot")
            return
        
        leaderboard = self.create_leaderboard()
        
        # Select metrics to plot
        metrics_to_plot = [
            'Service Level', 'Utilization Rate', 'Skill Coverage', 
            'Hiring Efficiency', 'Retention Rate', 'Demand Satisfaction'
        ]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in leaderboard.columns:
                ax = axes[i]
                bars = ax.bar(leaderboard['Model'], leaderboard[metric])
                ax.set_title(f'{metric}')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report.
        
        Returns:
            Evaluation report as string
        """
        if not self.metrics_history:
            return "No evaluation data available."
        
        leaderboard = self.create_leaderboard()
        
        report = []
        report.append("WORKFORCE PLANNING MODEL EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append(f"Number of models evaluated: {len(leaderboard)}")
        report.append(f"Best performing model: {leaderboard.iloc[0]['Model']}")
        report.append(f"Best composite score: {leaderboard.iloc[0]['Composite Score']:.3f}")
        report.append("")
        
        # Top 3 models
        report.append("TOP 3 MODELS:")
        for i in range(min(3, len(leaderboard))):
            model = leaderboard.iloc[i]
            report.append(f"{i+1}. {model['Model']} (Score: {model['Composite Score']:.3f})")
        report.append("")
        
        # Detailed metrics
        report.append("DETAILED METRICS:")
        for _, row in leaderboard.iterrows():
            report.append(f"\n{row['Model']}:")
            report.append(f"  Total Cost: ${row['Total Cost']:,.2f}")
            report.append(f"  Service Level: {row['Service Level']:.1%}")
            report.append(f"  Utilization Rate: {row['Utilization Rate']:.1%}")
            report.append(f"  Skill Coverage: {row['Skill Coverage']:.1%}")
            report.append(f"  Hiring Efficiency: {row['Hiring Efficiency']:.1%}")
            report.append(f"  Retention Rate: {row['Retention Rate']:.1%}")
            report.append(f"  Overtime Hours: {row['Overtime Hours']:.1f}")
            report.append(f"  Skill Gaps: {row['Skill Gaps']:.3f}")
            report.append(f"  Cost per Employee: ${row['Cost per Employee']:,.2f}")
        
        return "\n".join(report)


def main():
    """Test the evaluator."""
    evaluator = WorkforceEvaluator()
    
    # Create sample data for testing
    sample_result = {
        'success': True,
        'total_cost': 150000,
        'hires': {'Q1': 10, 'Q2': 15, 'Q3': 12},
        'method': 'test'
    }
    
    sample_demand = pd.DataFrame({
        'quarter': ['Q1', 'Q2', 'Q3'],
        'department': ['Engineering', 'Engineering', 'Engineering'],
        'demand': [50, 60, 55]
    })
    
    sample_employees = pd.DataFrame({
        'employee_id': ['EMP_001', 'EMP_002'],
        'department': ['Engineering', 'Engineering'],
        'skill_proficiencies': [
            {'Python': 0.9, 'SQL': 0.8},
            {'Python': 0.7, 'SQL': 0.9}
        ],
        'retention_probability': [0.9, 0.85]
    })
    
    sample_requirements = pd.DataFrame({
        'department': ['Engineering', 'Engineering'],
        'skill': ['Python', 'SQL'],
        'required_level': [0.8, 0.7]
    })
    
    # Evaluate
    metrics = evaluator.evaluate_model(
        sample_result, sample_demand, sample_employees, 
        sample_requirements, "test_model"
    )
    
    print("Sample Evaluation Metrics:")
    print(f"Total Cost: ${metrics.total_cost:,.2f}")
    print(f"Service Level: {metrics.service_level:.1%}")
    print(f"Utilization Rate: {metrics.utilization_rate:.1%}")
    print(f"Skill Coverage: {metrics.skill_coverage:.1%}")


if __name__ == "__main__":
    main()
