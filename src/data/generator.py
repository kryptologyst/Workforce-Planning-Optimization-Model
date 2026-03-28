"""Data generation and management for workforce planning model."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkforceDataConfig:
    """Configuration for synthetic workforce data generation."""
    n_quarters: int = 12
    n_departments: int = 5
    n_skills: int = 10
    n_employees: int = 200
    seed: int = 42


class WorkforceDataGenerator:
    """Generate synthetic workforce planning data."""
    
    def __init__(self, config: WorkforceDataConfig):
        """Initialize data generator with configuration.
        
        Args:
            config: Configuration for data generation
        """
        self.config = config
        np.random.seed(config.seed)
        
        # Department names
        self.departments = [
            "Engineering", "Sales", "Marketing", "Operations", "Support"
        ]
        
        # Skill categories
        self.skills = [
            "Python", "JavaScript", "SQL", "Project Management", 
            "Data Analysis", "Communication", "Leadership", 
            "Customer Service", "Design", "DevOps"
        ]
    
    def generate_demand_forecast(self) -> pd.DataFrame:
        """Generate demand forecast for each department and quarter.
        
        Returns:
            DataFrame with columns: quarter, department, demand, confidence
        """
        quarters = [f"Q{i//3 + 1}_{2024 + i//4}" for i in range(self.config.n_quarters)]
        
        data = []
        for quarter in quarters:
            for dept in self.departments:
                # Base demand with seasonal patterns
                base_demand = np.random.randint(20, 80)
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * len(data) / 4)
                demand = int(base_demand * seasonal_factor)
                confidence = np.random.uniform(0.7, 0.95)
                
                data.append({
                    "quarter": quarter,
                    "department": dept,
                    "demand": demand,
                    "confidence": confidence
                })
        
        return pd.DataFrame(data)
    
    def generate_employee_data(self) -> pd.DataFrame:
        """Generate current employee data with skills and attributes.
        
        Returns:
            DataFrame with employee information
        """
        employees = []
        
        for i in range(self.config.n_employees):
            # Basic employee info
            employee_id = f"EMP_{i+1:04d}"
            department = np.random.choice(self.departments)
            tenure_months = np.random.randint(1, 120)
            salary = np.random.randint(50000, 150000)
            
            # Skill proficiency (0-1 scale)
            skill_proficiencies = {}
            n_skills = np.random.randint(3, 8)
            selected_skills = np.random.choice(
                self.skills, size=n_skills, replace=False
            )
            
            for skill in selected_skills:
                skill_proficiencies[skill] = np.random.uniform(0.3, 1.0)
            
            # Performance and retention probability
            performance_score = np.random.uniform(0.6, 1.0)
            retention_prob = min(0.95, 0.7 + 0.2 * performance_score + 0.1 * (tenure_months / 120))
            
            employees.append({
                "employee_id": employee_id,
                "department": department,
                "tenure_months": tenure_months,
                "salary": salary,
                "performance_score": performance_score,
                "retention_probability": retention_prob,
                "skill_proficiencies": skill_proficiencies
            })
        
        return pd.DataFrame(employees)
    
    def generate_hiring_costs(self) -> pd.DataFrame:
        """Generate hiring costs by department and quarter.
        
        Returns:
            DataFrame with hiring cost information
        """
        quarters = [f"Q{i//3 + 1}_{2024 + i//4}" for i in range(self.config.n_quarters)]
        
        data = []
        for quarter in quarters:
            for dept in self.departments:
                # Base hiring cost with inflation
                base_cost = np.random.randint(8000, 15000)
                inflation_factor = 1 + 0.02 * len(data)  # 2% quarterly inflation
                hiring_cost = int(base_cost * inflation_factor)
                
                # Training cost (additional)
                training_cost = np.random.randint(2000, 5000)
                
                data.append({
                    "quarter": quarter,
                    "department": dept,
                    "hiring_cost": hiring_cost,
                    "training_cost": training_cost,
                    "total_cost": hiring_cost + training_cost
                })
        
        return pd.DataFrame(data)
    
    def generate_skill_requirements(self) -> pd.DataFrame:
        """Generate skill requirements for each department.
        
        Returns:
            DataFrame with skill requirements by department
        """
        data = []
        
        for dept in self.departments:
            # Each department has different skill requirements
            if dept == "Engineering":
                required_skills = ["Python", "JavaScript", "SQL", "DevOps"]
            elif dept == "Sales":
                required_skills = ["Communication", "Customer Service", "Project Management"]
            elif dept == "Marketing":
                required_skills = ["Data Analysis", "Communication", "Design"]
            elif dept == "Operations":
                required_skills = ["Project Management", "Leadership", "Data Analysis"]
            else:  # Support
                required_skills = ["Customer Service", "Communication", "Data Analysis"]
            
            for skill in self.skills:
                required_level = 0.8 if skill in required_skills else 0.3
                data.append({
                    "department": dept,
                    "skill": skill,
                    "required_level": required_level,
                    "is_critical": skill in required_skills
                })
        
        return pd.DataFrame(data)
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all workforce planning datasets.
        
        Returns:
            Dictionary containing all generated datasets
        """
        logger.info("Generating synthetic workforce planning data...")
        
        data = {
            "demand_forecast": self.generate_demand_forecast(),
            "employees": self.generate_employee_data(),
            "hiring_costs": self.generate_hiring_costs(),
            "skill_requirements": self.generate_skill_requirements()
        }
        
        logger.info(f"Generated {len(data)} datasets with {sum(len(df) for df in data.values())} total records")
        return data
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "data/raw") -> None:
        """Save generated data to CSV files.
        
        Args:
            data: Dictionary of DataFrames to save
            output_dir: Directory to save files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data.items():
            filepath = f"{output_dir}/{name}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} to {filepath}")


def main():
    """Generate and save synthetic workforce data."""
    config = WorkforceDataConfig()
    generator = WorkforceDataGenerator(config)
    
    data = generator.generate_all_data()
    generator.save_data(data)
    
    # Print summary
    print("Generated Workforce Planning Data:")
    for name, df in data.items():
        print(f"\n{name}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head())


if __name__ == "__main__":
    main()
