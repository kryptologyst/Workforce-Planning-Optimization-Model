"""Test suite for workforce planning model."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.data.generator import WorkforceDataGenerator, WorkforceDataConfig
from src.optimization.workforce_optimizer import WorkforceOptimizer, OptimizationConfig
from src.eval.metrics import WorkforceEvaluator, EvaluationMetrics
from src.viz.plots import WorkforceVisualizer
from src.utils.helpers import validate_data, calculate_summary_stats, format_currency


class TestWorkforceDataGenerator:
    """Test workforce data generator."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = WorkforceDataConfig()
        assert config.n_quarters == 12
        assert config.n_departments == 5
        assert config.n_employees == 200
        assert config.seed == 42
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        config = WorkforceDataConfig(n_quarters=4, n_departments=2)
        generator = WorkforceDataGenerator(config)
        
        assert generator.config == config
        assert len(generator.departments) == 2
        assert len(generator.skills) == 10
    
    def test_demand_forecast_generation(self):
        """Test demand forecast generation."""
        config = WorkforceDataConfig(n_quarters=4, n_departments=2)
        generator = WorkforceDataGenerator(config)
        
        forecast = generator.generate_demand_forecast()
        
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 8  # 4 quarters * 2 departments
        assert 'quarter' in forecast.columns
        assert 'department' in forecast.columns
        assert 'demand' in forecast.columns
        assert 'confidence' in forecast.columns
        assert all(forecast['demand'] > 0)
    
    def test_employee_data_generation(self):
        """Test employee data generation."""
        config = WorkforceDataConfig(n_employees=10)
        generator = WorkforceDataGenerator(config)
        
        employees = generator.generate_employee_data()
        
        assert isinstance(employees, pd.DataFrame)
        assert len(employees) == 10
        assert 'employee_id' in employees.columns
        assert 'department' in employees.columns
        assert 'skill_proficiencies' in employees.columns
        assert 'retention_probability' in employees.columns
    
    def test_hiring_costs_generation(self):
        """Test hiring costs generation."""
        config = WorkforceDataConfig(n_quarters=4, n_departments=2)
        generator = WorkforceDataGenerator(config)
        
        costs = generator.generate_hiring_costs()
        
        assert isinstance(costs, pd.DataFrame)
        assert len(costs) == 8  # 4 quarters * 2 departments
        assert 'quarter' in costs.columns
        assert 'department' in costs.columns
        assert 'total_cost' in costs.columns
        assert all(costs['total_cost'] > 0)
    
    def test_skill_requirements_generation(self):
        """Test skill requirements generation."""
        config = WorkforceDataConfig(n_departments=2)
        generator = WorkforceDataGenerator(config)
        
        requirements = generator.generate_skill_requirements()
        
        assert isinstance(requirements, pd.DataFrame)
        assert len(requirements) == 20  # 2 departments * 10 skills
        assert 'department' in requirements.columns
        assert 'skill' in requirements.columns
        assert 'required_level' in requirements.columns
        assert 'is_critical' in requirements.columns
    
    def test_all_data_generation(self):
        """Test complete data generation."""
        config = WorkforceDataConfig(n_quarters=4, n_departments=2, n_employees=10)
        generator = WorkforceDataGenerator(config)
        
        data = generator.generate_all_data()
        
        assert isinstance(data, dict)
        assert 'demand_forecast' in data
        assert 'employees' in data
        assert 'hiring_costs' in data
        assert 'skill_requirements' in data
        
        # Validate each dataset
        for key, df in data.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0


class TestWorkforceOptimizer:
    """Test workforce optimizer."""
    
    def test_config_initialization(self):
        """Test optimization configuration."""
        config = OptimizationConfig()
        assert config.solver == "highs"
        assert config.time_limit == 300
        assert config.max_hiring_per_quarter == 50
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = OptimizationConfig()
        optimizer = WorkforceOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.results == {}
    
    def test_basic_optimization(self):
        """Test basic optimization model."""
        config = OptimizationConfig()
        optimizer = WorkforceOptimizer(config)
        
        demand = [50, 60, 55]
        hiring_costs = [1000, 1100, 1200]
        
        result = optimizer.optimize_basic_model(demand, hiring_costs)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'method' in result
        
        if result['success']:
            assert 'hires' in result
            assert 'total_cost' in result
            assert isinstance(result['hires'], np.ndarray)
            assert result['total_cost'] > 0
    
    def test_cvxpy_optimization(self):
        """Test CVXPY optimization model."""
        config = OptimizationConfig()
        optimizer = WorkforceOptimizer(config)
        
        demand = [50, 60, 55]
        hiring_costs = [1000, 1100, 1200]
        
        result = optimizer.optimize_with_cvxpy(demand, hiring_costs)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'method' in result
    
    def test_method_comparison(self):
        """Test method comparison functionality."""
        config = OptimizationConfig()
        optimizer = WorkforceOptimizer(config)
        
        # Run basic optimization
        demand = [50, 60, 55]
        hiring_costs = [1000, 1100, 1200]
        optimizer.optimize_basic_model(demand, hiring_costs)
        
        comparison = optimizer.compare_methods()
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0
        assert 'method' in comparison.columns
    
    def test_optimal_plan_selection(self):
        """Test optimal plan selection."""
        config = OptimizationConfig()
        optimizer = WorkforceOptimizer(config)
        
        # Run optimization
        demand = [50, 60, 55]
        hiring_costs = [1000, 1100, 1200]
        optimizer.optimize_basic_model(demand, hiring_costs)
        
        optimal = optimizer.get_optimal_plan()
        
        assert isinstance(optimal, dict)
        assert 'success' in optimal


class TestWorkforceEvaluator:
    """Test workforce evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = WorkforceEvaluator()
        
        assert evaluator.metrics_history == []
        assert evaluator.baseline_metrics is None
    
    def test_service_level_calculation(self):
        """Test service level calculation."""
        evaluator = WorkforceEvaluator()
        
        actual_staffing = {'dept1': 50, 'dept2': 40}
        demand = {'dept1': 60, 'dept2': 30}
        
        service_level = evaluator.calculate_service_level(actual_staffing, demand)
        
        assert isinstance(service_level, float)
        assert 0 <= service_level <= 1
    
    def test_utilization_rate_calculation(self):
        """Test utilization rate calculation."""
        evaluator = WorkforceEvaluator()
        
        employees = pd.DataFrame({
            'department': ['Engineering', 'Sales'],
            'skill_proficiencies': [{}, {}]
        })
        
        demand_forecast = pd.DataFrame({
            'quarter': ['Q1', 'Q1'],
            'department': ['Engineering', 'Sales'],
            'demand': [50, 30]
        })
        
        utilization = evaluator.calculate_utilization_rate(employees, demand_forecast)
        
        assert isinstance(utilization, float)
        assert 0 <= utilization <= 1
    
    def test_skill_coverage_calculation(self):
        """Test skill coverage calculation."""
        evaluator = WorkforceEvaluator()
        
        employees = pd.DataFrame({
            'department': ['Engineering'],
            'skill_proficiencies': [{'Python': 0.9, 'SQL': 0.8}]
        })
        
        skill_requirements = pd.DataFrame({
            'department': ['Engineering', 'Engineering'],
            'skill': ['Python', 'SQL'],
            'required_level': [0.8, 0.7]
        })
        
        coverage = evaluator.calculate_skill_coverage(employees, skill_requirements)
        
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1
    
    def test_leaderboard_creation(self):
        """Test leaderboard creation."""
        evaluator = WorkforceEvaluator()
        
        # Add sample metrics
        evaluator.metrics_history.append({
            'model': 'test_model',
            'metrics': EvaluationMetrics(
                total_cost=100000,
                service_level=0.9,
                utilization_rate=0.8,
                skill_coverage=0.85,
                hiring_efficiency=0.75,
                retention_rate=0.9,
                overtime_hours=10,
                skill_gaps=0.1,
                cost_per_employee=50000,
                demand_satisfaction=0.9
            ),
            'timestamp': pd.Timestamp.now()
        })
        
        leaderboard = evaluator.create_leaderboard()
        
        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == 1
        assert 'Model' in leaderboard.columns
        assert 'Composite Score' in leaderboard.columns


class TestWorkforceVisualizer:
    """Test workforce visualizer."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = WorkforceVisualizer()
        
        assert visualizer.theme == "plotly_white"
        assert visualizer.color_palette == "viridis"
    
    def test_demand_forecast_plot(self):
        """Test demand forecast plotting."""
        visualizer = WorkforceVisualizer()
        
        demand_data = pd.DataFrame({
            'quarter': ['Q1', 'Q2'],
            'department': ['Engineering', 'Engineering'],
            'demand': [50, 60]
        })
        
        fig = visualizer.plot_demand_forecast(demand_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_hiring_plan_plot(self):
        """Test hiring plan plotting."""
        visualizer = WorkforceVisualizer()
        
        result = {
            'success': True,
            'hires': {'Q1': 10, 'Q2': 15}
        }
        
        fig = visualizer.plot_hiring_plan(result)
        
        assert fig is not None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_data_validation(self):
        """Test data validation."""
        valid_data = {
            'demand_forecast': pd.DataFrame({
                'quarter': ['Q1'],
                'department': ['Engineering'],
                'demand': [50]
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
        
        assert validate_data(valid_data) == True
    
    def test_invalid_data_validation(self):
        """Test invalid data validation."""
        invalid_data = {
            'demand_forecast': pd.DataFrame({
                'quarter': ['Q1'],
                'department': ['Engineering']
                # Missing 'demand' column
            })
        }
        
        assert validate_data(invalid_data) == False
    
    def test_summary_stats_calculation(self):
        """Test summary statistics calculation."""
        data = {
            'demand_forecast': pd.DataFrame({
                'quarter': ['Q1', 'Q2'],
                'department': ['Engineering', 'Engineering'],
                'demand': [50, 60]
            }),
            'employees': pd.DataFrame({
                'employee_id': ['EMP_001'],
                'department': ['Engineering'],
                'tenure_months': [24],
                'salary': [80000],
                'performance_score': [0.9],
                'retention_probability': [0.85],
                'skill_proficiencies': [{'Python': 0.9}]
            })
        }
        
        stats = calculate_summary_stats(data)
        
        assert isinstance(stats, dict)
        assert 'demand' in stats
        assert 'employees' in stats
    
    def test_currency_formatting(self):
        """Test currency formatting."""
        assert format_currency(123456.78) == "$123,456.78"
        assert format_currency(1000) == "$1,000.00"
    
    def test_percentage_formatting(self):
        """Test percentage formatting."""
        assert format_percentage(0.8567) == "85.7%"
        assert format_percentage(0.5, 0) == "50%"


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        config = WorkforceDataConfig(n_quarters=4, n_departments=2, n_employees=10)
        generator = WorkforceDataGenerator(config)
        data = generator.generate_all_data()
        
        # Validate data
        assert validate_data(data)
        
        # Run optimization
        opt_config = OptimizationConfig()
        optimizer = WorkforceOptimizer(opt_config)
        
        demand = data['demand_forecast'].groupby('quarter')['demand'].sum().tolist()
        hiring_costs = data['hiring_costs'].groupby('quarter')['total_cost'].mean().tolist()
        
        result = optimizer.optimize_basic_model(demand, hiring_costs)
        
        # Evaluate results
        evaluator = WorkforceEvaluator()
        if result['success']:
            metrics = evaluator.evaluate_model(
                result,
                data['demand_forecast'],
                data['employees'],
                data['skill_requirements'],
                "test_model"
            )
            
            assert isinstance(metrics, EvaluationMetrics)
            assert metrics.total_cost > 0
        
        # Create visualizations
        visualizer = WorkforceVisualizer()
        if result['success']:
            fig = visualizer.plot_hiring_plan(result)
            assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__])
