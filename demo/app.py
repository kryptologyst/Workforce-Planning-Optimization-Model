"""Interactive Streamlit demo for workforce planning optimization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.generator import WorkforceDataGenerator, WorkforceDataConfig
from src.optimization.workforce_optimizer import WorkforceOptimizer, OptimizationConfig
from src.eval.metrics import WorkforceEvaluator
from src.viz.plots import WorkforceVisualizer


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Workforce Planning Optimization",
        page_icon="👥",
        layout="wide"
    )
    
    # Header
    st.title("👥 Workforce Planning Optimization Model")
    st.markdown("""
    **DISCLAIMER**: This is an experimental research/educational tool. 
    Do not use for automated workforce decisions without human review and validation.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    n_quarters = st.sidebar.slider("Number of Quarters", 4, 12, 8)
    n_departments = st.sidebar.slider("Number of Departments", 3, 8, 5)
    n_employees = st.sidebar.slider("Current Employees", 100, 500, 200)
    seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)
    
    # Optimization parameters
    st.sidebar.subheader("Optimization Parameters")
    solver = st.sidebar.selectbox("Solver", ["highs", "cbc", "gurobi"], index=0)
    max_hiring = st.sidebar.slider("Max Hiring per Quarter", 10, 100, 50)
    budget_limit = st.sidebar.number_input("Budget Limit ($)", 50000, 2000000, 1000000)
    min_retention = st.sidebar.slider("Min Retention Rate", 0.7, 0.95, 0.85)
    
    # Generate data
    if st.sidebar.button("Generate New Data"):
        st.session_state.data_generated = False
    
    if 'data_generated' not in st.session_state or not st.session_state.data_generated:
        with st.spinner("Generating synthetic workforce data..."):
            config = WorkforceDataConfig(
                n_quarters=n_quarters,
                n_departments=n_departments,
                n_employees=n_employees,
                seed=seed
            )
            
            generator = WorkforceDataGenerator(config)
            data = generator.generate_all_data()
            
            st.session_state.data = data
            st.session_state.data_generated = True
        
        st.success("Data generated successfully!")
    
    data = st.session_state.data
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🎯 Optimization", 
        "📈 Results", 
        "🔍 Analysis", 
        "📋 Report"
    ])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demand Forecast")
            st.dataframe(data['demand_forecast'], use_container_width=True)
            
            # Plot demand forecast
            visualizer = WorkforceVisualizer()
            fig_demand = visualizer.plot_demand_forecast(data['demand_forecast'])
            st.plotly_chart(fig_demand, use_container_width=True)
        
        with col2:
            st.subheader("Current Employees")
            st.dataframe(data['employees'].head(10), use_container_width=True)
            
            # Employee distribution
            dept_counts = data['employees']['department'].value_counts()
            fig_dept = go.Figure(data=[
                go.Bar(x=dept_counts.index, y=dept_counts.values)
            ])
            fig_dept.update_layout(
                title="Employee Distribution by Department",
                xaxis_title="Department",
                yaxis_title="Number of Employees"
            )
            st.plotly_chart(fig_dept, use_container_width=True)
        
        st.subheader("Hiring Costs")
        st.dataframe(data['hiring_costs'], use_container_width=True)
        
        st.subheader("Skill Requirements")
        st.dataframe(data['skill_requirements'], use_container_width=True)
    
    with tab2:
        st.header("Workforce Optimization")
        
        # Optimization configuration
        opt_config = OptimizationConfig(
            solver=solver,
            max_hiring_per_quarter=max_hiring,
            budget_limit=budget_limit,
            min_retention_rate=min_retention
        )
        
        optimizer = WorkforceOptimizer(opt_config)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Model")
            if st.button("Run Basic Optimization"):
                with st.spinner("Running basic optimization..."):
                    # Extract demand and costs for basic model
                    demand = data['demand_forecast'].groupby('quarter')['demand'].sum().tolist()
                    hiring_costs = data['hiring_costs'].groupby('quarter')['total_cost'].mean().tolist()
                    
                    result = optimizer.optimize_basic_model(demand, hiring_costs)
                    
                    if result['success']:
                        st.success("Basic optimization completed!")
                        st.json(result)
                    else:
                        st.error(f"Optimization failed: {result.get('message', 'Unknown error')}")
        
        with col2:
            st.subheader("Advanced Model")
            if st.button("Run Advanced Optimization"):
                with st.spinner("Running advanced optimization..."):
                    result = optimizer.optimize_advanced_model(
                        data['demand_forecast'],
                        data['employees'],
                        data['hiring_costs'],
                        data['skill_requirements']
                    )
                    
                    if result['success']:
                        st.success("Advanced optimization completed!")
                        st.json(result)
                    else:
                        st.error(f"Optimization failed: {result.get('message', 'Unknown error')}")
        
        # CVXPY comparison
        st.subheader("CVXPY Comparison")
        if st.button("Run CVXPY Optimization"):
            with st.spinner("Running CVXPY optimization..."):
                demand = data['demand_forecast'].groupby('quarter')['demand'].sum().tolist()
                hiring_costs = data['hiring_costs'].groupby('quarter')['total_cost'].mean().tolist()
                
                result = optimizer.optimize_with_cvxpy(demand, hiring_costs)
                
                if result['success']:
                    st.success("CVXPY optimization completed!")
                    st.json(result)
                else:
                    st.error(f"Optimization failed: {result.get('message', 'Unknown error')}")
        
        # Method comparison
        if optimizer.results:
            st.subheader("Method Comparison")
            comparison_df = optimizer.compare_methods()
            st.dataframe(comparison_df, use_container_width=True)
    
    with tab3:
        st.header("Optimization Results")
        
        if not optimizer.results:
            st.warning("Please run optimization first.")
        else:
            # Get optimal result
            optimal = optimizer.get_optimal_plan()
            
            if optimal['success']:
                st.success(f"Optimal Plan Found (Method: {optimal['method']})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Cost", f"${optimal['total_cost']:,.2f}")
                    st.metric("Method", optimal['method'])
                
                with col2:
                    if 'hires' in optimal:
                        if isinstance(optimal['hires'], dict):
                            total_hires = sum(optimal['hires'].values())
                        else:
                            total_hires = sum(optimal['hires'])
                        st.metric("Total Hires", total_hires)
                
                # Plot hiring plan
                visualizer = WorkforceVisualizer()
                fig_hiring = visualizer.plot_hiring_plan(optimal)
                st.plotly_chart(fig_hiring, use_container_width=True)
                
                # Cost analysis
                fig_cost = visualizer.plot_cost_analysis(optimal, data['hiring_costs'])
                st.plotly_chart(fig_cost, use_container_width=True)
            else:
                st.error(f"No optimal solution found: {optimal.get('message', 'Unknown error')}")
    
    with tab4:
        st.header("Performance Analysis")
        
        if not optimizer.results:
            st.warning("Please run optimization first.")
        else:
            # Evaluate models
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
            
            # Create leaderboard
            leaderboard = evaluator.create_leaderboard()
            if not leaderboard.empty:
                st.subheader("Model Performance Leaderboard")
                st.dataframe(leaderboard, use_container_width=True)
                
                # Plot metrics comparison
                visualizer = WorkforceVisualizer()
                fig_metrics = visualizer.plot_metrics_dashboard(evaluator.metrics_history)
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Skill analysis
            st.subheader("Skill Analysis")
            fig_skills = visualizer.plot_skill_analysis(data['employees'], data['skill_requirements'])
            st.plotly_chart(fig_skills, use_container_width=True)
    
    with tab5:
        st.header("Evaluation Report")
        
        if not optimizer.results:
            st.warning("Please run optimization first.")
        else:
            evaluator = WorkforceEvaluator()
            
            # Evaluate all successful models
            for method, result in optimizer.results.items():
                if result['success']:
                    evaluator.evaluate_model(
                        result,
                        data['demand_forecast'],
                        data['employees'],
                        data['skill_requirements'],
                        method
                    )
            
            # Generate report
            report = evaluator.generate_evaluation_report()
            st.text(report)
            
            # Download report
            st.download_button(
                label="Download Report",
                data=report,
                file_name="workforce_planning_report.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Important Notes:**
    - This tool is for research and educational purposes only
    - All workforce decisions should involve human review and validation
    - Results are based on synthetic data and simplified models
    - Consider additional factors like market conditions, company culture, and legal requirements
    """)


if __name__ == "__main__":
    main()
