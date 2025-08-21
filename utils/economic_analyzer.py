import numpy as np

class EconomicAnalyzer:
    """Analyzer for economic viability and financial impact of waste management strategies"""
    
    def __init__(self):
        # Market prices (USD per kg)
        self.market_prices = {
            'plastic': {
                'clean': 0.65,
                'contaminated': 0.25,
                'mixed': 0.35
            },
            'metal': {
                'clean': 2.40,
                'contaminated': 1.80,
                'mixed': 2.10
            },
            'paper': {
                'clean': 0.18,
                'contaminated': 0.05,
                'mixed': 0.12
            },
            'glass': {
                'clean': 0.12,
                'contaminated': 0.08,
                'mixed': 0.10
            },
            'electronics': {
                'clean': 4.50,
                'contaminated': 2.80,
                'mixed': 3.20
            },
            'textile': {
                'clean': 0.35,
                'contaminated': 0.15,
                'mixed': 0.25
            }
        }
        
        # Processing costs (USD per kg)
        self.processing_costs = {
            'reuse': 0.05,      # Cleaning and preparation
            'recycle': 0.25,    # Processing and transformation
            'upcycle': 0.40,    # Creative processing and design
            'dispose': 0.12     # Disposal fees
        }
        
        # Market demand multipliers
        self.demand_multipliers = {
            'plastic': 1.2,     # High demand
            'metal': 1.5,       # Very high demand
            'paper': 0.8,       # Declining demand
            'glass': 0.9,       # Stable demand
            'electronics': 1.4, # High value components
            'textile': 0.6      # Limited recycling market
        }
    
    def calculate_value(self, material_type, action, quantity_kg, contamination_percent=0):
        """Calculate economic value of a waste management action"""
        if material_type not in self.market_prices:
            return 0.0
        
        # Determine contamination category
        if contamination_percent <= 10:
            condition = 'clean'
        elif contamination_percent <= 40:
            condition = 'mixed'
        else:
            condition = 'contaminated'
        
        base_price = self.market_prices[material_type][condition]
        processing_cost = self.processing_costs[action]
        demand_multiplier = self.demand_multipliers[material_type]
        
        # Calculate gross value
        gross_value = base_price * quantity_kg * demand_multiplier
        
        # Apply action-specific adjustments
        if action == 'reuse':
            # Higher value due to avoided processing
            adjusted_value = gross_value * 1.3
        elif action == 'recycle':
            # Standard recycling value
            adjusted_value = gross_value * 1.0
        elif action == 'upcycle':
            # Higher value but higher processing costs
            adjusted_value = gross_value * 1.6
        elif action == 'dispose':
            # Disposal cost (negative value)
            adjusted_value = -processing_cost * quantity_kg
            return adjusted_value
        else:
            adjusted_value = gross_value
        
        # Subtract processing costs
        net_value = adjusted_value - (processing_cost * quantity_kg)
        
        return max(0, net_value)  # Ensure non-negative value
    
    def calculate_roi(self, material_type, action, quantity_kg, contamination_percent, investment_cost=0):
        """Calculate return on investment for a waste management strategy"""
        value = self.calculate_value(material_type, action, quantity_kg, contamination_percent)
        
        if investment_cost == 0:
            # Use default investment costs
            investment_cost = self._estimate_investment_cost(action, quantity_kg)
        
        if investment_cost > 0:
            roi = ((value - investment_cost) / investment_cost) * 100
        else:
            roi = float('inf') if value > 0 else 0
        
        return roi
    
    def _estimate_investment_cost(self, action, quantity_kg):
        """Estimate initial investment cost for different actions"""
        base_costs = {
            'reuse': 50,        # Cleaning equipment, storage
            'recycle': 200,     # Processing equipment setup
            'upcycle': 500,     # Design and specialized equipment
            'dispose': 25       # Transport and disposal prep
        }
        
        # Scale with quantity (diminishing returns)
        scale_factor = np.log(max(1, quantity_kg)) / np.log(100)  # Normalized to 100kg
        
        return base_costs[action] * scale_factor
    
    def get_market_trends(self, material_type):
        """Get market trend data for a material"""
        trends = {
            'plastic': {
                'price_trend': 'increasing',
                'demand_trend': 'increasing',
                'volatility': 'medium',
                '6_month_outlook': 'positive',
                'key_factors': ['Plastic ban policies', 'Circular economy initiatives']
            },
            'metal': {
                'price_trend': 'stable',
                'demand_trend': 'stable',
                'volatility': 'low',
                '6_month_outlook': 'stable',
                'key_factors': ['Construction demand', 'Infrastructure projects']
            },
            'paper': {
                'price_trend': 'decreasing',
                'demand_trend': 'decreasing',
                'volatility': 'medium',
                '6_month_outlook': 'negative',
                'key_factors': ['Digital transformation', 'Packaging shift to plastic']
            },
            'glass': {
                'price_trend': 'stable',
                'demand_trend': 'stable',
                'volatility': 'low',
                '6_month_outlook': 'stable',
                'key_factors': ['Beverage industry demand', 'Construction glass']
            },
            'electronics': {
                'price_trend': 'increasing',
                'demand_trend': 'increasing',
                'volatility': 'high',
                '6_month_outlook': 'positive',
                'key_factors': ['Rare earth scarcity', 'E-waste regulations']
            },
            'textile': {
                'price_trend': 'developing',
                'demand_trend': 'increasing',
                'volatility': 'high',
                '6_month_outlook': 'positive',
                'key_factors': ['Fast fashion concerns', 'Sustainable fashion growth']
            }
        }
        
        return trends.get(material_type, {})
    
    def compare_strategies_economically(self, material_type, quantity_kg, contamination_percent):
        """Compare economic performance of different strategies"""
        strategies = ['reuse', 'recycle', 'upcycle', 'dispose']
        comparison = {}
        
        for strategy in strategies:
            value = self.calculate_value(material_type, strategy, quantity_kg, contamination_percent)
            roi = self.calculate_roi(material_type, strategy, quantity_kg, contamination_percent)
            
            comparison[strategy] = {
                'total_value': value,
                'value_per_kg': value / quantity_kg if quantity_kg > 0 else 0,
                'roi_percent': roi,
                'payback_period_months': self._calculate_payback_period(strategy, value, quantity_kg)
            }
        
        return comparison
    
    def _calculate_payback_period(self, action, net_value, quantity_kg):
        """Calculate payback period in months"""
        investment = self._estimate_investment_cost(action, quantity_kg)
        
        if net_value > 0 and investment > 0:
            monthly_return = net_value / 12  # Assume annual returns spread over 12 months
            payback_months = investment / monthly_return if monthly_return > 0 else float('inf')
            return min(payback_months, 60)  # Cap at 5 years
        
        return float('inf')
    
    def get_cost_breakdown(self, material_type, action, quantity_kg, contamination_percent):
        """Get detailed cost breakdown"""
        condition = 'clean' if contamination_percent <= 10 else 'mixed' if contamination_percent <= 40 else 'contaminated'
        
        breakdown = {
            'material_value': self.market_prices[material_type][condition] * quantity_kg,
            'processing_cost': self.processing_costs[action] * quantity_kg,
            'transport_cost': 0.03 * quantity_kg,  # Estimated transport
            'overhead_cost': 0.02 * quantity_kg,   # Administrative overhead
            'market_adjustment': self.demand_multipliers[material_type]
        }
        
        breakdown['gross_revenue'] = breakdown['material_value'] * breakdown['market_adjustment']
        breakdown['total_costs'] = breakdown['processing_cost'] + breakdown['transport_cost'] + breakdown['overhead_cost']
        breakdown['net_profit'] = breakdown['gross_revenue'] - breakdown['total_costs']
        breakdown['profit_margin'] = (breakdown['net_profit'] / breakdown['gross_revenue'] * 100) if breakdown['gross_revenue'] > 0 else 0
        
        return breakdown
    
    def calculate_scale_economies(self, material_type, action, base_quantity_kg):
        """Calculate how economics change with scale"""
        scales = [base_quantity_kg * multiplier for multiplier in [0.5, 1, 2, 5, 10]]
        scale_analysis = {}
        
        for scale in scales:
            value = self.calculate_value(material_type, action, scale, 20)  # Assume 20% contamination
            cost_per_kg = (self.processing_costs[action] * (0.9 ** np.log(scale/base_quantity_kg)))  # Economies of scale
            
            scale_analysis[f"{scale:.0f}kg"] = {
                'total_value': value,
                'value_per_kg': value / scale,
                'processing_cost_per_kg': cost_per_kg,
                'net_value_per_kg': (value / scale) - cost_per_kg
            }
        
        return scale_analysis
