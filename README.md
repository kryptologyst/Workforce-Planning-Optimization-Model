# Workforce Planning Optimization Model

A comprehensive workforce planning optimization system that helps determine optimal hiring strategies across departments and time periods using advanced linear programming and constraint optimization techniques.

## DISCLAIMER

**IMPORTANT**: This is an experimental research and educational tool. Do not use for automated workforce decisions without human review and validation. All workforce planning decisions should involve human oversight and consider additional factors such as market conditions, company culture, legal requirements, and strategic objectives.

## Overview

This project provides a modernized workforce planning model that extends the basic linear programming approach with advanced features including:

- **Multi-department optimization** with skill-based constraints
- **Attrition modeling** and retention rate considerations
- **Budget constraints** and hiring limits
- **Skill gap analysis** and coverage optimization
- **Multiple optimization solvers** (SciPy, PuLP, CVXPY)
- **Comprehensive evaluation metrics** and performance comparison
- **Interactive visualization** and reporting tools

## Features

### Core Capabilities
- **Demand Forecasting**: Generate synthetic demand patterns with seasonal variations
- **Employee Modeling**: Track skills, performance, and retention probabilities
- **Cost Optimization**: Minimize total workforce costs while meeting demand
- **Constraint Handling**: Budget limits, hiring caps, retention requirements
- **Multi-Solver Support**: Compare different optimization approaches

### Advanced Features
- **Skill-Based Planning**: Match employee skills to department requirements
- **Attrition Modeling**: Account for employee turnover in planning
- **Overtime Optimization**: Balance hiring costs with overtime expenses
- **Performance Metrics**: Service level, utilization, skill coverage analysis
- **Interactive Demo**: Streamlit-based web application for exploration

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Setup
```bash
# Clone or download the project
cd workforce-planning-model

# Install dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Quick Start

### 1. Generate Data and Run Optimization
```bash
python main.py
```

This will:
- Generate synthetic workforce data
- Run multiple optimization models
- Evaluate performance metrics
- Create visualizations and reports
- Save results to `assets/` directory

### 2. Interactive Demo
```bash
streamlit run demo/app.py
```

Launch the interactive web application to:
- Explore data and configurations
- Run different optimization scenarios
- Visualize results and metrics
- Generate custom reports

### 3. Custom Configuration
Edit `configs/config.yaml` to modify:
- Data generation parameters
- Optimization constraints
- Evaluation metrics
- Visualization settings

## Project Structure

```
workforce-planning-model/
├── src/                          # Source code
│   ├── data/                     # Data generation and management
│   │   └── generator.py          # Synthetic data generation
│   ├── optimization/             # Optimization models
│   │   └── workforce_optimizer.py # Core optimization logic
│   ├── eval/                     # Evaluation and metrics
│   │   └── metrics.py           # Performance evaluation
│   ├── viz/                      # Visualization tools
│   │   └── plots.py             # Plotting functions
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
│   └── config.yaml              # Main configuration
├── demo/                         # Interactive demo
│   └── app.py                   # Streamlit application
├── data/                         # Data storage
│   ├── raw/                      # Raw generated data
│   └── processed/                # Processed data
├── assets/                       # Generated outputs
│   ├── *.html                    # Interactive plots
│   ├── *.txt                     # Reports
│   └── *.pkl                     # Results
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── notebooks/                     # Jupyter notebooks
├── logs/                         # Log files
├── main.py                       # Main execution script
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## Data Schema

### Input Data
- **demand_forecast.csv**: Quarterly demand by department
- **employees.csv**: Current employee data with skills and attributes
- **hiring_costs.csv**: Hiring costs by quarter and department
- **skill_requirements.csv**: Required skills by department

### Generated Data
- **transactions.csv**: Employee transactions and movements
- **workforce_calendar.csv**: Workforce availability and scheduling
- **performance_metrics.csv**: Employee performance data

## Optimization Models

### 1. Basic Linear Programming
- Simple quarterly hiring optimization
- Minimize total hiring costs
- Meet cumulative demand constraints
- Uses SciPy's `linprog` solver

### 2. Advanced Constraint Optimization
- Multi-department planning
- Skill-based constraints
- Attrition and retention modeling
- Budget and hiring limits
- Overtime optimization
- Uses PuLP solver

### 3. CVXPY Optimization
- Convex optimization approach
- Advanced constraint handling
- Multiple solver options
- Comparison with other methods

## Evaluation Metrics

### Primary Metrics
- **Total Cost**: Overall workforce cost
- **Service Level**: Demand satisfaction rate
- **Utilization Rate**: Workforce efficiency
- **Skill Coverage**: Skill requirement fulfillment

### Secondary Metrics
- **Hiring Efficiency**: Cost per hire effectiveness
- **Retention Rate**: Employee retention performance
- **Overtime Hours**: Overtime usage
- **Skill Gaps**: Unmet skill requirements

### Business KPIs
- **Cost per Employee**: Average cost per headcount
- **Demand Satisfaction**: Percentage of demand met
- **Composite Score**: Weighted performance metric

## Configuration

### Data Parameters
```yaml
data:
  synthetic:
    n_quarters: 12          # Planning horizon
    n_departments: 5        # Number of departments
    n_skills: 10            # Skill categories
    n_employees: 200        # Current workforce size
    seed: 42                # Random seed
```

### Optimization Settings
```yaml
model:
  optimization:
    solver: "highs"         # Solver choice
    time_limit: 300         # Time limit (seconds)
    gap_tolerance: 0.01     # Optimality gap
```

### Business Constraints
```yaml
constraints:
  max_hiring_per_quarter: 50    # Hiring limit
  min_retention_rate: 0.85       # Retention requirement
  max_overtime_hours: 20         # Overtime limit
  skill_match_threshold: 0.8     # Skill requirement
  budget_limit: 1000000          # Budget constraint
```

## Usage Examples

### Basic Usage
```python
from src.data.generator import WorkforceDataGenerator, WorkforceDataConfig
from src.optimization.workforce_optimizer import WorkforceOptimizer, OptimizationConfig

# Generate data
config = WorkforceDataConfig()
generator = WorkforceDataGenerator(config)
data = generator.generate_all_data()

# Run optimization
opt_config = OptimizationConfig()
optimizer = WorkforceOptimizer(opt_config)
result = optimizer.optimize_advanced_model(
    data['demand_forecast'],
    data['employees'],
    data['hiring_costs'],
    data['skill_requirements']
)
```

### Custom Evaluation
```python
from src.eval.metrics import WorkforceEvaluator

evaluator = WorkforceEvaluator()
metrics = evaluator.evaluate_model(
    result,
    data['demand_forecast'],
    data['employees'],
    data['skill_requirements'],
    "custom_model"
)

print(f"Total Cost: ${metrics.total_cost:,.2f}")
print(f"Service Level: {metrics.service_level:.1%}")
```

### Visualization
```python
from src.viz.plots import WorkforceVisualizer

visualizer = WorkforceVisualizer()
fig = visualizer.plot_hiring_plan(result)
fig.show()
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Development

### Code Quality
- **Formatting**: Black for code formatting
- **Linting**: Ruff for code linting
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google/NumPy docstring format

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Limitations and Considerations

### Model Limitations
- **Synthetic Data**: Results based on generated data, not real workforce
- **Simplified Constraints**: Real-world constraints may be more complex
- **Static Planning**: Does not account for dynamic market changes
- **Skill Modeling**: Simplified skill representation

### Business Considerations
- **Human Oversight**: All decisions require human review
- **Legal Compliance**: Consider employment laws and regulations
- **Cultural Factors**: Account for company culture and values
- **Market Conditions**: External factors not modeled
- **Strategic Alignment**: Ensure alignment with business strategy

### Technical Considerations
- **Solver Limitations**: Different solvers have different capabilities
- **Scalability**: Performance may vary with problem size
- **Data Quality**: Results depend on input data accuracy
- **Model Assumptions**: Verify assumptions match business reality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed description
4. Include error messages and system information

## Changelog

### Version 1.0.0
- Initial release with modernized workforce planning model
- Multi-solver optimization support
- Comprehensive evaluation metrics
- Interactive Streamlit demo
- Advanced constraint handling
- Skill-based planning capabilities

## Acknowledgments

- Built on top of SciPy, PuLP, CVXPY optimization libraries
- Visualization powered by Plotly and Matplotlib
- Interactive interface using Streamlit
- Configuration management with OmegaConf
# Workforce-Planning-Optimization-Model
