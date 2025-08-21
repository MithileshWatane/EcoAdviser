import numpy as np

class CarbonCalculator:
    """Calculator for environmental impact and carbon footprint analysis"""
    
    def __init__(self):
        # Carbon emissions data (kg CO2 per kg of material)
        self.emission_factors = {
            'production': {
                'plastic': 3.4,  # kg CO2 per kg
                'metal': 5.8,   # Average for aluminum
                'paper': 1.1,
                'glass': 0.85,
                'electronics': 12.5,  # High due to complex manufacturing
                'textile': 2.8
            },
            'disposal': {
                'plastic': 2.1,  # Landfill emissions
                'metal': 0.3,   # Minimal disposal impact
                'paper': -0.4,  # Carbon sequestration in landfill
                'glass': 0.1,   # Minimal impact
                'electronics': 4.5,  # Hazardous disposal
                'textile': 1.2
            },
            'recycling_process': {
                'plastic': 0.8,  # Energy for processing
                'metal': 1.2,   # Smelting energy
                'paper': 0.9,   # Pulping process
                'glass': 0.5,   # Melting energy
                'electronics': 2.3,  # Complex processing
                'textile': 1.5
            }
        }
        
        # Savings from avoided production (per kg recycled)
        self.recycling_benefits = {
            'plastic': 2.6,  # CO2 saved by not producing new plastic
            'metal': 4.6,    # High savings for metals
            'paper': 0.7,
            'glass': 0.35,
            'electronics': 10.2,  # High value components
            'textile': 1.3
        }
    
    def calculate_savings(self, material_type, action, quantity_kg):
        """Calculate carbon savings for a specific action"""
        if material_type not in self.emission_factors['production']:
            return 0.0
        
        baseline_impact = self._calculate_baseline_impact(material_type, quantity_kg)
        action_impact = self._calculate_action_impact(material_type, action, quantity_kg)
        
        savings = baseline_impact - action_impact
        return max(0, savings)  # Ensure non-negative savings
    
    def _calculate_baseline_impact(self, material_type, quantity_kg):
        """Calculate baseline impact (disposal + replacement production)"""
        disposal_impact = self.emission_factors['disposal'][material_type] * quantity_kg
        replacement_impact = self.emission_factors['production'][material_type] * quantity_kg
        
        return disposal_impact + replacement_impact
    
    def _calculate_action_impact(self, material_type, action, quantity_kg):
        """Calculate impact of taking a specific action"""
        if action == 'dispose':
            return self.emission_factors['disposal'][material_type] * quantity_kg + \
                   self.emission_factors['production'][material_type] * quantity_kg
        
        elif action == 'reuse':
            # Minimal processing, high benefit
            return 0.1 * quantity_kg  # Small cleaning/transport impact
        
        elif action == 'recycle':
            processing_impact = self.emission_factors['recycling_process'][material_type] * quantity_kg
            avoided_production = self.recycling_benefits[material_type] * quantity_kg
            return processing_impact - avoided_production
        
        elif action == 'upcycle':
            # Slightly higher processing than recycling
            processing_impact = self.emission_factors['recycling_process'][material_type] * quantity_kg * 1.3
            avoided_production = self.recycling_benefits[material_type] * quantity_kg * 0.8
            return processing_impact - avoided_production
        
        return 0.0
    
    def get_disposal_impact(self, material_type, quantity_kg):
        """Get the impact of disposal alone"""
        if material_type not in self.emission_factors['disposal']:
            return 0.0
        
        return self.emission_factors['disposal'][material_type] * quantity_kg
    
    def calculate_equivalent_metrics(self, carbon_saved_kg):
        """Convert carbon savings to equivalent metrics"""
        equivalents = {
            'trees_planted': carbon_saved_kg / 22,  # 1 tree absorbs ~22kg CO2/year
            'car_miles_offset': carbon_saved_kg * 2.3,  # ~0.43kg CO2 per mile
            'home_energy_days': carbon_saved_kg / 28,  # ~28kg CO2 per day for average home
            'plastic_bottles_recycled': carbon_saved_kg / 0.082  # ~0.082kg CO2 per bottle
        }
        
        return equivalents
    
    def get_monthly_projection(self, material_type, action, daily_quantity_kg):
        """Project monthly savings based on daily waste"""
        monthly_quantity = daily_quantity_kg * 30
        return self.calculate_savings(material_type, action, monthly_quantity)
    
    def get_annual_projection(self, material_type, action, daily_quantity_kg):
        """Project annual savings based on daily waste"""
        annual_quantity = daily_quantity_kg * 365
        return self.calculate_savings(material_type, action, annual_quantity)
    
    def compare_strategies(self, material_type, quantity_kg):
        """Compare carbon impact of different strategies"""
        strategies = ['reuse', 'recycle', 'upcycle', 'dispose']
        comparison = {}
        
        for strategy in strategies:
            savings = self.calculate_savings(material_type, strategy, quantity_kg)
            comparison[strategy] = {
                'carbon_saved': savings,
                'impact_per_kg': savings / quantity_kg if quantity_kg > 0 else 0
            }
        
        return comparison
    
    def get_industry_benchmarks(self, material_type):
        """Get industry benchmarks for carbon performance"""
        benchmarks = {
            'plastic': {
                'best_practice_saving': 2.8,  # kg CO2 per kg material
                'average_saving': 1.8,
                'poor_practice_saving': 0.5
            },
            'metal': {
                'best_practice_saving': 4.2,
                'average_saving': 3.5,
                'poor_practice_saving': 1.2
            },
            'paper': {
                'best_practice_saving': 0.8,
                'average_saving': 0.5,
                'poor_practice_saving': 0.1
            },
            'glass': {
                'best_practice_saving': 0.4,
                'average_saving': 0.25,
                'poor_practice_saving': 0.05
            },
            'electronics': {
                'best_practice_saving': 8.5,
                'average_saving': 6.2,
                'poor_practice_saving': 2.1
            },
            'textile': {
                'best_practice_saving': 1.8,
                'average_saving': 1.0,
                'poor_practice_saving': 0.2
            }
        }
        
        return benchmarks.get(material_type, {})
    
    def calculate_lifecycle_impact(self, material_type, quantity_kg, use_cycles=1):
        """Calculate impact considering multiple use cycles"""
        single_cycle_production = self.emission_factors['production'][material_type] * quantity_kg
        
        # Each reuse cycle avoids new production
        total_avoided_production = single_cycle_production * (use_cycles - 1)
        
        # Small impact for each reuse (cleaning, transport)
        reuse_impacts = 0.05 * quantity_kg * (use_cycles - 1)
        
        net_benefit = total_avoided_production - reuse_impacts
        
        return max(0, net_benefit)
