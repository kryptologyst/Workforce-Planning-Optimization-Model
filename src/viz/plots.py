"""Visualization tools for workforce planning analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class WorkforceVisualizer:
    """Visualization tools for workforce planning."""
    
    def __init__(self, theme: str = "plotly_white", color_palette: str = "viridis"):
        """Initialize visualizer with theme and color settings.
        
        Args:
            theme: Plotly theme
            color_palette: Color palette for plots
        """
        self.theme = theme
        self.color_palette = color_palette
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(color_palette)
    
    def plot_demand_forecast(
        self, 
        demand_forecast: pd.DataFrame,
        title: str = "Workforce Demand Forecast",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot demand forecast by department and quarter.
        
        Args:
            demand_forecast: Demand forecast data
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        fig = px.line(
            demand_forecast, 
            x='quarter', 
            y='demand', 
            color='department',
            title=title,
            template=self.theme
        )
        
        fig.update_layout(
            xaxis_title="Quarter",
            yaxis_title="Demand (Employees)",
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Demand forecast plot saved to {save_path}")
        
        return fig
    
    def plot_hiring_plan(
        self,
        optimization_result: Dict[str, Any],
        title: str = "Optimal Hiring Plan",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot optimal hiring plan.
        
        Args:
            optimization_result: Optimization results
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        if 'hires' not in optimization_result:
            logger.warning("No hiring data in optimization result")
            return go.Figure()
        
        hires = optimization_result['hires']
        
        if isinstance(hires, dict) and len(hires) > 0:
            # Advanced model with department-wise data
            if isinstance(list(hires.keys())[0], tuple):
                # (quarter, department) format
                data = []
                for (quarter, dept), count in hires.items():
                    data.append({
                        'quarter': quarter,
                        'department': dept,
                        'hires': count
                    })
                df = pd.DataFrame(data)
                
                fig = px.bar(
                    df,
                    x='quarter',
                    y='hires',
                    color='department',
                    title=title,
                    template=self.theme
                )
            else:
                # Simple quarterly format
                quarters = list(hires.keys())
                counts = list(hires.values())
                
                fig = go.Figure(data=[
                    go.Bar(x=quarters, y=counts, name='Hires')
                ])
                fig.update_layout(
                    title=title,
                    xaxis_title="Quarter",
                    yaxis_title="Number of Hires",
                    template=self.theme
                )
        else:
            # Basic model with array
            quarters = [f"Q{i+1}" for i in range(len(hires))]
            fig = go.Figure(data=[
                go.Bar(x=quarters, y=hires, name='Hires')
            ])
            fig.update_layout(
                title=title,
                xaxis_title="Quarter",
                yaxis_title="Number of Hires",
                template=self.theme
            )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Hiring plan plot saved to {save_path}")
        
        return fig
    
    def plot_cost_analysis(
        self,
        optimization_result: Dict[str, Any],
        hiring_costs: pd.DataFrame,
        title: str = "Cost Analysis",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot cost analysis breakdown.
        
        Args:
            optimization_result: Optimization results
            hiring_costs: Hiring cost data
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        if 'hires' not in optimization_result:
            logger.warning("No hiring data for cost analysis")
            return go.Figure()
        
        hires = optimization_result['hires']
        total_cost = optimization_result.get('total_cost', 0)
        
        # Calculate cost breakdown
        cost_breakdown = []
        
        if isinstance(hires, dict) and len(hires) > 0:
            if isinstance(list(hires.keys())[0], tuple):
                # (quarter, department) format
                for (quarter, dept), count in hires.items():
                    cost_data = hiring_costs[
                        (hiring_costs['quarter'] == quarter) & 
                        (hiring_costs['department'] == dept)
                    ]
                    if not cost_data.empty:
                        cost_per_hire = cost_data.iloc[0]['total_cost']
                        total_quarter_cost = count * cost_per_hire
                        cost_breakdown.append({
                            'quarter': quarter,
                            'department': dept,
                            'cost': total_quarter_cost,
                            'hires': count
                        })
            else:
                # Simple quarterly format
                for quarter, count in hires.items():
                    cost_data = hiring_costs[hiring_costs['quarter'] == quarter]
                    if not cost_data.empty:
                        avg_cost = cost_data['total_cost'].mean()
                        total_quarter_cost = count * avg_cost
                        cost_breakdown.append({
                            'quarter': quarter,
                            'cost': total_quarter_cost,
                            'hires': count
                        })
        
        if not cost_breakdown:
            logger.warning("No cost breakdown data available")
            return go.Figure()
        
        df = pd.DataFrame(cost_breakdown)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cost by Quarter', 'Hires by Quarter'),
            vertical_spacing=0.1
        )
        
        # Cost plot
        if 'department' in df.columns:
            for dept in df['department'].unique():
                dept_data = df[df['department'] == dept]
                fig.add_trace(
                    go.Bar(
                        x=dept_data['quarter'],
                        y=dept_data['cost'],
                        name=f'{dept} Cost',
                        showlegend=True
                    ),
                    row=1, col=1
                )
        else:
            fig.add_trace(
                go.Bar(
                    x=df['quarter'],
                    y=df['cost'],
                    name='Cost',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Hires plot
        if 'department' in df.columns:
            for dept in df['department'].unique():
                dept_data = df[df['department'] == dept]
                fig.add_trace(
                    go.Bar(
                        x=dept_data['quarter'],
                        y=dept_data['hires'],
                        name=f'{dept} Hires',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        else:
            fig.add_trace(
                go.Bar(
                    x=df['quarter'],
                    y=df['hires'],
                    name='Hires',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600
        )
        
        fig.update_xaxes(title_text="Quarter", row=2, col=1)
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Hires", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Cost analysis plot saved to {save_path}")
        
        return fig
    
    def plot_skill_analysis(
        self,
        employees: pd.DataFrame,
        skill_requirements: pd.DataFrame,
        title: str = "Skill Analysis",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot skill analysis across departments.
        
        Args:
            employees: Employee data with skills
            skill_requirements: Skill requirements by department
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        # Calculate skill levels by department
        skill_data = []
        
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
                skill_gap = max(0, required_level - avg_skill_level)
            else:
                avg_skill_level = 0
                skill_gap = required_level
            
            skill_data.append({
                'department': dept,
                'skill': skill,
                'current_level': avg_skill_level,
                'required_level': required_level,
                'skill_gap': skill_gap
            })
        
        df = pd.DataFrame(skill_data)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Current vs Required Skill Levels', 'Skill Gaps'),
            horizontal_spacing=0.1
        )
        
        # Skill levels comparison
        for skill in df['skill'].unique():
            skill_data = df[df['skill'] == skill]
            
            fig.add_trace(
                go.Bar(
                    x=skill_data['department'],
                    y=skill_data['current_level'],
                    name=f'{skill} (Current)',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=skill_data['department'],
                    y=skill_data['required_level'],
                    name=f'{skill} (Required)',
                    marker_color='darkblue'
                ),
                row=1, col=1
            )
        
        # Skill gaps
        for skill in df['skill'].unique():
            skill_data = df[df['skill'] == skill]
            
            fig.add_trace(
                go.Bar(
                    x=skill_data['department'],
                    y=skill_data['skill_gap'],
                    name=f'{skill} Gap',
                    marker_color='red'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=500
        )
        
        fig.update_xaxes(title_text="Department", row=1, col=1)
        fig.update_xaxes(title_text="Department", row=1, col=2)
        fig.update_yaxes(title_text="Skill Level", row=1, col=1)
        fig.update_yaxes(title_text="Skill Gap", row=1, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Skill analysis plot saved to {save_path}")
        
        return fig
    
    def plot_metrics_dashboard(
        self,
        evaluation_metrics: List[Dict[str, Any]],
        title: str = "Performance Metrics Dashboard",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create a comprehensive metrics dashboard.
        
        Args:
            evaluation_metrics: List of evaluation metrics
            title: Dashboard title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        if not evaluation_metrics:
            logger.warning("No metrics data for dashboard")
            return go.Figure()
        
        # Extract metrics
        models = [m['model'] for m in evaluation_metrics]
        metrics_data = {
            'Service Level': [m['metrics'].service_level for m in evaluation_metrics],
            'Utilization Rate': [m['metrics'].utilization_rate for m in evaluation_metrics],
            'Skill Coverage': [m['metrics'].skill_coverage for m in evaluation_metrics],
            'Hiring Efficiency': [m['metrics'].hiring_efficiency for m in evaluation_metrics],
            'Retention Rate': [m['metrics'].retention_rate for m in evaluation_metrics]
        }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=list(metrics_data.keys()),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Add bar charts for each metric
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            row, col = positions[i]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric_name,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Add composite score scatter plot
        composite_scores = []
        for m in evaluation_metrics:
            # Calculate composite score
            score = (
                m['metrics'].service_level * 0.25 +
                m['metrics'].utilization_rate * 0.20 +
                m['metrics'].skill_coverage * 0.15 +
                m['metrics'].hiring_efficiency * 0.15 +
                m['metrics'].retention_rate * 0.10 +
                m['metrics'].demand_satisfaction * 0.15
            )
            composite_scores.append(score)
        
        fig.add_trace(
            go.Scatter(
                x=models,
                y=composite_scores,
                mode='markers+lines',
                name='Composite Score',
                marker=dict(size=10),
                showlegend=False
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Metrics dashboard saved to {save_path}")
        
        return fig
    
    def create_sankey_diagram(
        self,
        employees: pd.DataFrame,
        title: str = "Employee Flow",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create a Sankey diagram for employee flow.
        
        Args:
            employees: Employee data
            title: Diagram title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        # Calculate flows between departments
        departments = employees['department'].unique()
        
        # For simplicity, create a circular flow
        source = []
        target = []
        value = []
        labels = list(departments)
        
        for i, dept in enumerate(departments):
            # Flow to next department (circular)
            next_dept_idx = (i + 1) % len(departments)
            source.append(i)
            target.append(next_dept_idx)
            
            # Value based on department size
            dept_size = len(employees[employees['department'] == dept])
            value.append(dept_size)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="blue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        fig.update_layout(
            title_text=title,
            font_size=10,
            template=self.theme
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Sankey diagram saved to {save_path}")
        
        return fig


def main():
    """Test the visualizer."""
    visualizer = WorkforceVisualizer()
    
    # Create sample data
    sample_demand = pd.DataFrame({
        'quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
        'department': ['Engineering', 'Engineering', 'Engineering', 'Engineering'],
        'demand': [50, 60, 55, 70]
    })
    
    sample_result = {
        'success': True,
        'total_cost': 150000,
        'hires': {'Q1': 10, 'Q2': 15, 'Q3': 12, 'Q4': 18}
    }
    
    # Test plots
    fig1 = visualizer.plot_demand_forecast(sample_demand)
    fig2 = visualizer.plot_hiring_plan(sample_result)
    
    print("Visualization tests completed")


if __name__ == "__main__":
    main()
