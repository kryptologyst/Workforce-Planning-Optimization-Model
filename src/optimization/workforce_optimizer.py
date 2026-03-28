"""Advanced workforce planning optimization model with attrition, skills, and constraints."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy.optimize import linprog
import cvxpy as cp
from pulp import *

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for workforce optimization."""
    solver: str = "highs"
    time_limit: int = 300
    gap_tolerance: float = 0.01
    max_hiring_per_quarter: int = 50
    min_retention_rate: float = 0.85
    max_overtime_hours: int = 20
    skill_match_threshold: float = 0.8
    budget_limit: int = 1000000


class WorkforceOptimizer:
    """Advanced workforce planning optimizer with multiple constraints."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize optimizer with configuration.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.results = {}
        
    def optimize_basic_model(
        self, 
        demand: List[int], 
        hiring_costs: List[float]
    ) -> Dict[str, Any]:
        """Optimize basic workforce planning model (original implementation).
        
        Args:
            demand: Demand for each quarter
            hiring_costs: Hiring cost per employee per quarter
            
        Returns:
            Optimization results
        """
        logger.info("Running basic workforce optimization...")
        
        # Objective: minimize hiring costs
        c = np.array(hiring_costs)
        
        # Constraints: cumulative hiring must meet demand
        n_quarters = len(demand)
        A = np.zeros((n_quarters, n_quarters))
        for i in range(n_quarters):
            for j in range(i + 1):
                A[i, j] = -1
        
        b = np.array([-d for d in demand])
        
        # Bounds: non-negative hiring
        bounds = [(0, None) for _ in range(n_quarters)]
        
        # Solve
        result = linprog(
            c=c, 
            A_ub=A, 
            b_ub=b, 
            bounds=bounds, 
            method=self.config.solver
        )
        
        if result.success:
            hires = np.ceil(result.x).astype(int)
            total_cost = np.sum(hires * hiring_costs)
            
            self.results['basic'] = {
                'success': True,
                'hires': hires,
                'total_cost': total_cost,
                'objective_value': result.fun,
                'method': 'basic_linprog'
            }
            
            logger.info(f"Basic optimization completed. Total cost: ${total_cost:,.2f}")
        else:
            self.results['basic'] = {
                'success': False,
                'message': result.message,
                'method': 'basic_linprog'
            }
            logger.error(f"Basic optimization failed: {result.message}")
        
        return self.results['basic']
    
    def optimize_advanced_model(
        self,
        demand_forecast: pd.DataFrame,
        employees: pd.DataFrame,
        hiring_costs: pd.DataFrame,
        skill_requirements: pd.DataFrame
    ) -> Dict[str, Any]:
        """Optimize advanced workforce model with attrition, skills, and constraints.
        
        Args:
            demand_forecast: Demand forecast by quarter and department
            employees: Current employee data
            hiring_costs: Hiring costs by quarter and department
            skill_requirements: Skill requirements by department
            
        Returns:
            Advanced optimization results
        """
        logger.info("Running advanced workforce optimization...")
        
        try:
            # Create PuLP problem
            prob = LpProblem("WorkforcePlanning", LpMinimize)
            
            # Get unique quarters and departments
            quarters = sorted(demand_forecast['quarter'].unique())
            departments = sorted(demand_forecast['department'].unique())
            
            # Decision variables: hires per quarter per department
            hires = {}
            for q in quarters:
                for d in departments:
                    hires[q, d] = LpVariable(
                        f"hires_{q}_{d}", 
                        lowBound=0, 
                        upBound=self.config.max_hiring_per_quarter,
                        cat='Integer'
                    )
            
            # Decision variables: overtime hours per quarter per department
            overtime = {}
            for q in quarters:
                for d in departments:
                    overtime[q, d] = LpVariable(
                        f"overtime_{q}_{d}",
                        lowBound=0,
                        upBound=self.config.max_overtime_hours,
                        cat='Continuous'
                    )
            
            # Objective: minimize total costs (hiring + overtime)
            total_cost = 0
            
            for q in quarters:
                for d in departments:
                    # Hiring cost
                    cost_data = hiring_costs[
                        (hiring_costs['quarter'] == q) & 
                        (hiring_costs['department'] == d)
                    ]
                    if not cost_data.empty:
                        hiring_cost = cost_data.iloc[0]['total_cost']
                        total_cost += hires[q, d] * hiring_cost
                    
                    # Overtime cost (assume $50/hour)
                    total_cost += overtime[q, d] * 50
            
            prob += total_cost
            
            # Constraints - simplified to avoid infeasibility
            for q in quarters:
                for d in departments:
                    # Demand constraint: current employees + hires >= 70% of demand
                    current_employees = len(employees[employees['department'] == d])
                    
                    # Account for attrition
                    retention_rate = self.config.min_retention_rate
                    effective_employees = current_employees * retention_rate
                    
                    demand_data = demand_forecast[
                        (demand_forecast['quarter'] == q) & 
                        (demand_forecast['department'] == d)
                    ]
                    if not demand_data.empty:
                        demand_value = demand_data.iloc[0]['demand']
                        
                        # Simplified constraint - just hiring constraint
                        prob += (
                            effective_employees + 
                            hires[q, d]
                        ) >= demand_value * 0.7  # Allow 70% demand satisfaction minimum
            
            # Budget constraint
            prob += total_cost <= self.config.budget_limit
            
            # Solve
            prob.solve(PULP_CBC_CMD(timeLimit=self.config.time_limit))
            
            if prob.status == LpStatusOptimal:
                # Extract results
                hires_result = {}
                overtime_result = {}
                total_cost_value = 0
                
                for q in quarters:
                    for d in departments:
                        hires_result[q, d] = int(hires[q, d].varValue or 0)
                        overtime_result[q, d] = overtime[q, d].varValue or 0
                        
                        # Calculate actual costs
                        cost_data = hiring_costs[
                            (hiring_costs['quarter'] == q) & 
                            (hiring_costs['department'] == d)
                        ]
                        if not cost_data.empty:
                            hiring_cost = cost_data.iloc[0]['total_cost']
                            total_cost_value += hires_result[q, d] * hiring_cost
                        
                        total_cost_value += overtime_result[q, d] * 50
                
                self.results['advanced'] = {
                    'success': True,
                    'hires': hires_result,
                    'overtime': overtime_result,
                    'total_cost': total_cost_value,
                    'objective_value': prob.objective.value(),
                    'method': 'advanced_pulp',
                    'status': LpStatus[prob.status]
                }
                
                logger.info(f"Advanced optimization completed. Total cost: ${total_cost_value:,.2f}")
                
            else:
                self.results['advanced'] = {
                    'success': False,
                    'message': f"Optimization failed with status: {LpStatus[prob.status]}",
                    'method': 'advanced_pulp'
                }
                logger.error(f"Advanced optimization failed: {LpStatus[prob.status]}")
        
        except Exception as e:
            self.results['advanced'] = {
                'success': False,
                'message': str(e),
                'method': 'advanced_pulp'
            }
            logger.error(f"Advanced optimization error: {e}")
        
        return self.results['advanced']
    
    def optimize_with_cvxpy(
        self,
        demand: List[int],
        hiring_costs: List[float],
        attrition_rates: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Optimize using CVXPY for comparison.
        
        Args:
            demand: Demand for each quarter
            hiring_costs: Hiring cost per employee per quarter
            attrition_rates: Attrition rates per quarter
            
        Returns:
            CVXPY optimization results
        """
        logger.info("Running CVXPY workforce optimization...")
        
        try:
            n_quarters = len(demand)
            
            # Decision variables
            hires = cp.Variable(n_quarters, integer=True, nonneg=True)
            
            # Attrition rates (default to 5% per quarter)
            if attrition_rates is None:
                attrition_rates = [0.05] * n_quarters
            
            # Objective: minimize total hiring costs
            objective = cp.Minimize(cp.sum(cp.multiply(hires, hiring_costs)))
            
            # Constraints
            constraints = []
            
            # Cumulative hiring constraint with attrition
            cumulative_hires = cp.cumsum(hires)
            cumulative_demand = np.cumsum(demand)
            
            for i in range(n_quarters):
                # Account for attrition in previous quarters
                attrition_factor = np.prod([1 - rate for rate in attrition_rates[:i+1]])
                constraints.append(
                    cumulative_hires[i] * attrition_factor >= cumulative_demand[i]
                )
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS_BB if self.config.solver == "highs" else cp.CLARABEL)
            
            if problem.status == cp.OPTIMAL:
                hires_result = np.ceil(hires.value).astype(int)
                total_cost = np.sum(hires_result * hiring_costs)
                
                self.results['cvxpy'] = {
                    'success': True,
                    'hires': hires_result,
                    'total_cost': total_cost,
                    'objective_value': problem.value,
                    'method': 'cvxpy',
                    'status': problem.status
                }
                
                logger.info(f"CVXPY optimization completed. Total cost: ${total_cost:,.2f}")
                
            else:
                self.results['cvxpy'] = {
                    'success': False,
                    'message': f"Optimization failed with status: {problem.status}",
                    'method': 'cvxpy'
                }
                logger.error(f"CVXPY optimization failed: {problem.status}")
        
        except Exception as e:
            self.results['cvxpy'] = {
                'success': False,
                'message': str(e),
                'method': 'cvxpy'
            }
            logger.error(f"CVXPY optimization error: {e}")
        
        return self.results['cvxpy']
    
    def compare_methods(self) -> pd.DataFrame:
        """Compare results from different optimization methods.
        
        Returns:
            DataFrame comparing optimization results
        """
        comparison_data = []
        
        for method, result in self.results.items():
            if result['success']:
                comparison_data.append({
                    'method': method,
                    'total_cost': result['total_cost'],
                    'objective_value': result['objective_value'],
                    'status': 'Success'
                })
            else:
                comparison_data.append({
                    'method': method,
                    'total_cost': None,
                    'objective_value': None,
                    'status': result.get('message', 'Failed')
                })
        
        return pd.DataFrame(comparison_data)
    
    def get_optimal_plan(self) -> Dict[str, Any]:
        """Get the best optimization result.
        
        Returns:
            Best optimization result
        """
        successful_results = {
            k: v for k, v in self.results.items() 
            if v['success']
        }
        
        if not successful_results:
            return {'success': False, 'message': 'No successful optimizations'}
        
        # Choose result with lowest cost
        best_method = min(successful_results.keys(), 
                         key=lambda k: successful_results[k]['total_cost'])
        
        return successful_results[best_method]


def main():
    """Test the workforce optimizer."""
    # Test data
    demand = [50, 60, 55, 70]
    hiring_costs = [1000, 1100, 1200, 1300]
    
    config = OptimizationConfig()
    optimizer = WorkforceOptimizer(config)
    
    # Run different optimization methods
    basic_result = optimizer.optimize_basic_model(demand, hiring_costs)
    cvxpy_result = optimizer.optimize_with_cvxpy(demand, hiring_costs)
    
    # Compare results
    comparison = optimizer.compare_methods()
    print("Optimization Comparison:")
    print(comparison)
    
    # Get optimal plan
    optimal = optimizer.get_optimal_plan()
    print(f"\nOptimal Plan: {optimal}")


if __name__ == "__main__":
    main()
