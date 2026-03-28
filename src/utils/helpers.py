"""Utility functions for workforce planning model."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to file.
    
    Args:
        results: Results dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif filepath.suffix == '.json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = convert_for_json(results)
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Loaded results dictionary
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Results loaded from {filepath}")
    return results


def convert_for_json(obj: Any) -> Any:
    """Convert object for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj


def validate_data(data: Dict[str, pd.DataFrame]) -> bool:
    """Validate input data for workforce planning.
    
    Args:
        data: Dictionary of DataFrames to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    required_keys = ['demand_forecast', 'employees', 'hiring_costs', 'skill_requirements']
    
    # Check required keys
    for key in required_keys:
        if key not in data:
            logger.error(f"Missing required data: {key}")
            return False
    
    # Validate demand forecast
    df = data['demand_forecast']
    required_cols = ['quarter', 'department', 'demand']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column in demand_forecast: {col}")
            return False
    
    # Validate employees
    df = data['employees']
    required_cols = ['employee_id', 'department', 'skill_proficiencies']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column in employees: {col}")
            return False
    
    # Validate hiring costs
    df = data['hiring_costs']
    required_cols = ['quarter', 'department', 'total_cost']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column in hiring_costs: {col}")
            return False
    
    # Validate skill requirements
    df = data['skill_requirements']
    required_cols = ['department', 'skill', 'required_level']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column in skill_requirements: {col}")
            return False
    
    logger.info("Data validation passed")
    return True


def calculate_summary_stats(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Calculate summary statistics for workforce data.
    
    Args:
        data: Dictionary of DataFrames
        
    Returns:
        Summary statistics dictionary
    """
    stats = {}
    
    # Demand forecast stats
    if 'demand_forecast' in data:
        df = data['demand_forecast']
        stats['demand'] = {
            'total_demand': df['demand'].sum(),
            'avg_demand_per_quarter': df.groupby('quarter')['demand'].sum().mean(),
            'max_demand_per_quarter': df.groupby('quarter')['demand'].sum().max(),
            'min_demand_per_quarter': df.groupby('quarter')['demand'].sum().min(),
            'departments': df['department'].nunique(),
            'quarters': df['quarter'].nunique()
        }
    
    # Employee stats
    if 'employees' in data:
        df = data['employees']
        stats['employees'] = {
            'total_employees': len(df),
            'departments': df['department'].nunique(),
            'avg_tenure_months': df['tenure_months'].mean(),
            'avg_salary': df['salary'].mean(),
            'avg_performance': df['performance_score'].mean(),
            'avg_retention_rate': df['retention_probability'].mean()
        }
    
    # Hiring cost stats
    if 'hiring_costs' in data:
        df = data['hiring_costs']
        stats['hiring_costs'] = {
            'avg_cost_per_hire': df['total_cost'].mean(),
            'max_cost_per_hire': df['total_cost'].max(),
            'min_cost_per_hire': df['total_cost'].min(),
            'total_budget_needed': df['total_cost'].sum()
        }
    
    # Skill requirements stats
    if 'skill_requirements' in data:
        df = data['skill_requirements']
        stats['skills'] = {
            'total_skills': df['skill'].nunique(),
            'critical_skills': df[df['is_critical']]['skill'].nunique(),
            'avg_required_level': df['required_level'].mean(),
            'max_required_level': df['required_level'].max()
        }
    
    return stats


def format_currency(amount: float) -> str:
    """Format currency amount for display.
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage value for display.
    
    Args:
        value: Value to format (0-1 range)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def create_output_directory(base_path: str = "assets") -> Path:
    """Create output directory with timestamp.
    
    Args:
        base_path: Base path for output directory
        
    Returns:
        Path to created directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def log_model_performance(
    model_name: str,
    metrics: Dict[str, float],
    logger: logging.Logger
) -> None:
    """Log model performance metrics.
    
    Args:
        model_name: Name of the model
        metrics: Performance metrics dictionary
        logger: Logger instance
    """
    logger.info(f"Model: {model_name}")
    for metric_name, value in metrics.items():
        if 'cost' in metric_name.lower():
            logger.info(f"  {metric_name}: {format_currency(value)}")
        elif 'rate' in metric_name.lower() or 'level' in metric_name.lower():
            logger.info(f"  {metric_name}: {format_percentage(value)}")
        else:
            logger.info(f"  {metric_name}: {value:.3f}")


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available.
    
    Returns:
        Dictionary mapping package names to availability
    """
    dependencies = {
        'numpy': False,
        'pandas': False,
        'scipy': False,
        'matplotlib': False,
        'plotly': False,
        'streamlit': False,
        'pulp': False,
        'cvxpy': False,
        'omegaconf': False
    }
    
    for package in dependencies.keys():
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    return dependencies


def print_dependency_status() -> None:
    """Print dependency status."""
    deps = check_dependencies()
    
    print("Dependency Status:")
    print("-" * 20)
    
    for package, available in deps.items():
        status = "✓" if available else "✗"
        print(f"{status} {package}")
    
    missing = [pkg for pkg, avail in deps.items() if not avail]
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))


def main():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test dependency check
    print_dependency_status()
    
    # Test data validation
    sample_data = {
        'demand_forecast': pd.DataFrame({
            'quarter': ['Q1', 'Q2'],
            'department': ['Engineering', 'Engineering'],
            'demand': [50, 60]
        }),
        'employees': pd.DataFrame({
            'employee_id': ['EMP_001'],
            'department': ['Engineering'],
            'skill_proficiencies': [{'Python': 0.9}]
        }),
        'hiring_costs': pd.DataFrame({
            'quarter': ['Q1'],
            'department': ['Engineering'],
            'total_cost': [10000]
        }),
        'skill_requirements': pd.DataFrame({
            'department': ['Engineering'],
            'skill': ['Python'],
            'required_level': [0.8]
        })
    }
    
    is_valid = validate_data(sample_data)
    print(f"Data validation: {'Passed' if is_valid else 'Failed'}")
    
    # Test summary stats
    stats = calculate_summary_stats(sample_data)
    print(f"Summary stats calculated: {len(stats)} categories")
    
    # Test formatting
    print(f"Currency format: {format_currency(123456.78)}")
    print(f"Percentage format: {format_percentage(0.8567)}")
    
    print("Utility function tests completed!")


if __name__ == "__main__":
    main()
