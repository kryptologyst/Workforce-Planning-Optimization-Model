"""Modernized Workforce Planning Model - Main execution script."""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.generator import WorkforceDataGenerator, WorkforceDataConfig
from src.optimization.workforce_optimizer import WorkforceOptimizer, OptimizationConfig
from src.eval.metrics import WorkforceEvaluator
from src.viz.plots import WorkforceVisualizer
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/workforce_planning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(config, resolve=True)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            'data': {'synthetic': {'n_quarters': 8, 'n_departments': 5, 'n_employees': 200, 'seed': 42}},
            'model': {'optimization': {'solver': 'highs', 'time_limit': 300, 'gap_tolerance': 0.01}},
            'constraints': {'max_hiring_per_quarter': 50, 'min_retention_rate': 0.85, 'budget_limit': 1000000}
        }


def run_workforce_planning(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run complete workforce planning pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Results dictionary
    """
    logger.info("Starting workforce planning pipeline...")
    
    # Create output directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Step 1: Generate synthetic data
    logger.info("Step 1: Generating synthetic workforce data...")
    data_config = WorkforceDataConfig(**config['data']['synthetic'])
    generator = WorkforceDataGenerator(data_config)
    data = generator.generate_all_data()
    
    # Save data
    generator.save_data(data, "data/raw")
    
    # Step 2: Run optimization models
    logger.info("Step 2: Running optimization models...")
    opt_config = OptimizationConfig(**config['model']['optimization'])
    optimizer = WorkforceOptimizer(opt_config)
    
    # Basic model
    demand = data['demand_forecast'].groupby('quarter')['demand'].sum().tolist()
    hiring_costs = data['hiring_costs'].groupby('quarter')['total_cost'].mean().tolist()
    
    basic_result = optimizer.optimize_basic_model(demand, hiring_costs)
    logger.info(f"Basic optimization: {'Success' if basic_result['success'] else 'Failed'}")
    
    # Advanced model
    advanced_result = optimizer.optimize_advanced_model(
        data['demand_forecast'],
        data['employees'],
        data['hiring_costs'],
        data['skill_requirements']
    )
    logger.info(f"Advanced optimization: {'Success' if advanced_result['success'] else 'Failed'}")
    
    # CVXPY model
    cvxpy_result = optimizer.optimize_with_cvxpy(demand, hiring_costs)
    logger.info(f"CVXPY optimization: {'Success' if cvxpy_result['success'] else 'Failed'}")
    
    # Step 3: Evaluate models
    logger.info("Step 3: Evaluating models...")
    evaluator = WorkforceEvaluator()
    
    for method, result in optimizer.results.items():
        if result['success']:
            metrics = evaluator.evaluate_model(
                result,
                data['demand_forecast'],
                data['employees'],
                data['skill_requirements'],
                method
            )
            logger.info(f"Evaluated {method}: Cost=${metrics.total_cost:,.2f}, Service Level={metrics.service_level:.1%}")
    
    # Step 4: Create visualizations
    logger.info("Step 4: Creating visualizations...")
    visualizer = WorkforceVisualizer()
    
    # Plot demand forecast
    fig_demand = visualizer.plot_demand_forecast(data['demand_forecast'])
    fig_demand.write_html("assets/demand_forecast.html")
    
    # Plot hiring plans for successful optimizations
    for method, result in optimizer.results.items():
        if result['success']:
            fig_hiring = visualizer.plot_hiring_plan(result, f"Hiring Plan - {method}")
            fig_hiring.write_html(f"assets/hiring_plan_{method}.html")
    
    # Plot skill analysis
    fig_skills = visualizer.plot_skill_analysis(data['employees'], data['skill_requirements'])
    fig_skills.write_html("assets/skill_analysis.html")
    
    # Plot metrics dashboard
    if evaluator.metrics_history:
        fig_dashboard = visualizer.plot_metrics_dashboard(evaluator.metrics_history)
        fig_dashboard.write_html("assets/metrics_dashboard.html")
    
    # Step 5: Generate report
    logger.info("Step 5: Generating evaluation report...")
    report = evaluator.generate_evaluation_report()
    
    with open("assets/evaluation_report.txt", "w") as f:
        f.write(report)
    
    # Get optimal plan
    optimal_plan = optimizer.get_optimal_plan()
    
    # Compile results
    results = {
        'data': data,
        'optimization_results': optimizer.results,
        'optimal_plan': optimal_plan,
        'evaluation_metrics': evaluator.metrics_history,
        'leaderboard': evaluator.create_leaderboard(),
        'report': report
    }
    
    logger.info("Workforce planning pipeline completed successfully!")
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print results summary.
    
    Args:
        results: Results dictionary
    """
    print("\n" + "="*60)
    print("WORKFORCE PLANNING MODEL - RESULTS SUMMARY")
    print("="*60)
    
    # Data summary
    data = results['data']
    print(f"\nData Generated:")
    print(f"  - Quarters: {len(data['demand_forecast']['quarter'].unique())}")
    print(f"  - Departments: {len(data['demand_forecast']['department'].unique())}")
    print(f"  - Current Employees: {len(data['employees'])}")
    print(f"  - Skills: {len(data['skill_requirements']['skill'].unique())}")
    
    # Optimization results
    print(f"\nOptimization Results:")
    for method, result in results['optimization_results'].items():
        if result['success']:
            print(f"  - {method.upper()}: Success (Cost: ${result['total_cost']:,.2f})")
        else:
            print(f"  - {method.upper()}: Failed ({result.get('message', 'Unknown error')})")
    
    # Optimal plan
    optimal = results['optimal_plan']
    if optimal['success']:
        print(f"\nOptimal Plan ({optimal['method']}):")
        print(f"  - Total Cost: ${optimal['total_cost']:,.2f}")
        if 'hires' in optimal:
            if isinstance(optimal['hires'], dict):
                total_hires = sum(optimal['hires'].values())
            else:
                total_hires = sum(optimal['hires'])
            print(f"  - Total Hires: {total_hires}")
    
    # Leaderboard
    leaderboard = results['leaderboard']
    if not leaderboard.empty:
        print(f"\nModel Performance Ranking:")
        for _, row in leaderboard.head(3).iterrows():
            print(f"  {row['Rank']}. {row['Model']} (Score: {row['Composite Score']:.3f})")
    
    print(f"\nAssets Generated:")
    print(f"  - Demand forecast plot: assets/demand_forecast.html")
    print(f"  - Skill analysis plot: assets/skill_analysis.html")
    print(f"  - Metrics dashboard: assets/metrics_dashboard.html")
    print(f"  - Evaluation report: assets/evaluation_report.txt")
    
    print(f"\nTo run the interactive demo:")
    print(f"  streamlit run demo/app.py")
    
    print("\n" + "="*60)


def main():
    """Main execution function."""
    print("Workforce Planning Model - Modernized Version")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Run workforce planning pipeline
    results = run_workforce_planning(config)
    
    # Print results
    print_results(results)
    
    # Save results to file
    import pickle
    with open("assets/results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    logger.info("Results saved to assets/results.pkl")


if __name__ == "__main__":
    main()
