import pandas as pd
import numpy as np

class MaterialDatabase:
    """Database containing material properties, recycling methods, and recommendations"""
    
    def __init__(self):
        self.materials = self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the comprehensive material database"""
        database = {
            'plastic': {
                'properties': {
                    'density': '0.9-1.4 g/cm³',
                    'melting_point': '100-260°C',
                    'recyclability': 'Variable by type',
                    'degradation_time': '20-1000 years',
                    'contamination_sensitivity': 'High'
                },
                'recycling_methods': [
                    'Mechanical recycling (shredding, melting)',
                    'Chemical recycling (depolymerization)',
                    'Energy recovery (incineration)',
                    'Pyrolysis (thermal decomposition)'
                ],
                'recommendations': [
                    {
                        'action': 'recycle',
                        'method': 'Mechanical processing',
                        'description': 'Clean, sort, shred, and melt plastic into pellets for new products',
                        'viability_score': 8.5,
                        'requirements': 'Clean material, proper sorting',
                        'market_demand': 'High for clean PET and HDPE'
                    },
                    {
                        'action': 'upcycle',
                        'method': '3D printing filament',
                        'description': 'Convert plastic waste into 3D printing filament',
                        'viability_score': 7.2,
                        'requirements': 'Clean, specific plastic types',
                        'market_demand': 'Growing market'
                    },
                    {
                        'action': 'reuse',
                        'method': 'Direct reuse',
                        'description': 'Clean and reuse containers for storage or other purposes',
                        'viability_score': 9.0,
                        'requirements': 'Food-safe cleaning',
                        'market_demand': 'Local/individual use'
                    },
                    {
                        'action': 'dispose',
                        'method': 'Waste-to-energy',
                        'description': 'Incineration with energy recovery',
                        'viability_score': 4.0,
                        'requirements': 'Proper incineration facility',
                        'market_demand': 'Energy market'
                    }
                ],
                'market_data': {
                    'price_per_kg': 0.45,
                    'demand_level': 'High',
                    'trend': 'Increasing'
                }
            },
            'metal': {
                'properties': {
                    'density': '2.7-11.3 g/cm³',
                    'melting_point': '660-1538°C',
                    'recyclability': 'Excellent',
                    'degradation_time': 'Does not degrade',
                    'contamination_sensitivity': 'Low'
                },
                'recycling_methods': [
                    'Smelting and refining',
                    'Magnetic separation',
                    'Eddy current separation',
                    'Density separation'
                ],
                'recommendations': [
                    {
                        'action': 'recycle',
                        'method': 'Smelting',
                        'description': 'Melt down metal and purify for new products',
                        'viability_score': 9.5,
                        'requirements': 'Sorting by metal type',
                        'market_demand': 'Very high'
                    },
                    {
                        'action': 'reuse',
                        'method': 'Structural reuse',
                        'description': 'Use metal components directly in construction or manufacturing',
                        'viability_score': 8.8,
                        'requirements': 'Structural integrity testing',
                        'market_demand': 'Construction industry'
                    },
                    {
                        'action': 'upcycle',
                        'method': 'Art and furniture',
                        'description': 'Transform into decorative or functional items',
                        'viability_score': 6.5,
                        'requirements': 'Creative design, welding skills',
                        'market_demand': 'Niche market'
                    }
                ],
                'market_data': {
                    'price_per_kg': 1.85,
                    'demand_level': 'Very High',
                    'trend': 'Stable'
                }
            },
            'paper': {
                'properties': {
                    'density': '0.7-1.15 g/cm³',
                    'degradation_time': '2-6 months',
                    'recyclability': 'Good (5-7 cycles)',
                    'contamination_sensitivity': 'Very high',
                    'fiber_length': 'Decreases with recycling'
                },
                'recycling_methods': [
                    'Pulping and de-inking',
                    'Flotation separation',
                    'Screening and cleaning',
                    'Bleaching and forming'
                ],
                'recommendations': [
                    {
                        'action': 'recycle',
                        'method': 'Pulping process',
                        'description': 'Break down into fibers and reform into new paper products',
                        'viability_score': 8.8,
                        'requirements': 'Clean, dry paper',
                        'market_demand': 'High'
                    },
                    {
                        'action': 'reuse',
                        'method': 'Direct reuse',
                        'description': 'Use for packaging, wrapping, or office use',
                        'viability_score': 9.2,
                        'requirements': 'Good condition',
                        'market_demand': 'Office and retail'
                    },
                    {
                        'action': 'upcycle',
                        'method': 'Insulation material',
                        'description': 'Process into cellulose insulation',
                        'viability_score': 7.8,
                        'requirements': 'Fire retardant treatment',
                        'market_demand': 'Construction'
                    },
                    {
                        'action': 'dispose',
                        'method': 'Composting',
                        'description': 'Biodegradation into soil amendment',
                        'viability_score': 6.0,
                        'requirements': 'Chemical-free paper',
                        'market_demand': 'Agriculture'
                    }
                ],
                'market_data': {
                    'price_per_kg': 0.12,
                    'demand_level': 'Medium',
                    'trend': 'Declining'
                }
            },
            'glass': {
                'properties': {
                    'density': '2.4-2.8 g/cm³',
                    'melting_point': '1700°C',
                    'recyclability': 'Infinite',
                    'degradation_time': '1 million years',
                    'contamination_sensitivity': 'Medium'
                },
                'recycling_methods': [
                    'Crushing and cullet formation',
                    'Color sorting',
                    'Contaminant removal',
                    'Melting and forming'
                ],
                'recommendations': [
                    {
                        'action': 'recycle',
                        'method': 'Closed-loop recycling',
                        'description': 'Crush, melt, and form into new glass products',
                        'viability_score': 9.8,
                        'requirements': 'Color separation',
                        'market_demand': 'High'
                    },
                    {
                        'action': 'reuse',
                        'method': 'Container reuse',
                        'description': 'Clean and reuse jars and bottles for storage',
                        'viability_score': 9.5,
                        'requirements': 'Thorough cleaning',
                        'market_demand': 'Food industry'
                    },
                    {
                        'action': 'upcycle',
                        'method': 'Aggregate substitute',
                        'description': 'Crush into aggregate for construction',
                        'viability_score': 7.5,
                        'requirements': 'Size grading',
                        'market_demand': 'Construction'
                    }
                ],
                'market_data': {
                    'price_per_kg': 0.08,
                    'demand_level': 'Medium',
                    'trend': 'Stable'
                }
            },
            'electronics': {
                'properties': {
                    'complexity': 'Very high',
                    'hazardous_materials': 'Heavy metals, rare earths',
                    'recyclability': 'Complex',
                    'degradation_time': '50-1000 years',
                    'value_density': 'Very high'
                },
                'recycling_methods': [
                    'Manual disassembly',
                    'Shredding and separation',
                    'Hydrometallurgy',
                    'Pyrometallurgy'
                ],
                'recommendations': [
                    {
                        'action': 'recycle',
                        'method': 'E-waste processing',
                        'description': 'Extract valuable metals and materials through specialized processing',
                        'viability_score': 8.2,
                        'requirements': 'Certified e-waste facility',
                        'market_demand': 'High for precious metals'
                    },
                    {
                        'action': 'reuse',
                        'method': 'Refurbishment',
                        'description': 'Repair and resell functional devices',
                        'viability_score': 9.0,
                        'requirements': 'Working condition, data wiping',
                        'market_demand': 'Secondary market'
                    },
                    {
                        'action': 'upcycle',
                        'method': 'Component harvesting',
                        'description': 'Extract functional components for repair or maker projects',
                        'viability_score': 6.8,
                        'requirements': 'Technical knowledge',
                        'market_demand': 'Maker community'
                    }
                ],
                'market_data': {
                    'price_per_kg': 3.50,
                    'demand_level': 'High',
                    'trend': 'Increasing'
                }
            },
            'textile': {
                'properties': {
                    'fiber_composition': 'Natural/synthetic blend',
                    'degradation_time': '6 months - 200 years',
                    'recyclability': 'Limited',
                    'contamination_sensitivity': 'High',
                    'processing_complexity': 'Medium'
                },
                'recycling_methods': [
                    'Mechanical recycling (shredding)',
                    'Chemical recycling (dissolution)',
                    'Thermal recycling',
                    'Biological treatment'
                ],
                'recommendations': [
                    {
                        'action': 'reuse',
                        'method': 'Clothing donation',
                        'description': 'Donate wearable items to extend their life',
                        'viability_score': 9.5,
                        'requirements': 'Good condition, clean',
                        'market_demand': 'Charity organizations'
                    },
                    {
                        'action': 'upcycle',
                        'method': 'Fashion upcycling',
                        'description': 'Transform into new clothing or accessories',
                        'viability_score': 7.8,
                        'requirements': 'Design skills, sewing',
                        'market_demand': 'Fashion market'
                    },
                    {
                        'action': 'recycle',
                        'method': 'Fiber recovery',
                        'description': 'Break down into fibers for new textiles',
                        'viability_score': 6.5,
                        'requirements': 'Specialized facility',
                        'market_demand': 'Textile industry'
                    },
                    {
                        'action': 'upcycle',
                        'method': 'Insulation material',
                        'description': 'Shred into building insulation',
                        'viability_score': 7.2,
                        'requirements': 'Fire retardant treatment',
                        'market_demand': 'Construction'
                    }
                ],
                'market_data': {
                    'price_per_kg': 0.25,
                    'demand_level': 'Low',
                    'trend': 'Developing'
                }
            }
        }
        return database
    
    def get_recommendations(self, material_type, contamination_level=0):
        """Get recommendations for a material type considering contamination"""
        if material_type not in self.materials:
            return []
        
        recommendations = self.materials[material_type]['recommendations'].copy()
        
        # Adjust viability scores based on contamination
        contamination_factor = 1 - (contamination_level / 100) * 0.3
        
        for rec in recommendations:
            rec['viability_score'] = min(10.0, rec['viability_score'] * contamination_factor)
        
        # Sort by viability score
        recommendations.sort(key=lambda x: x['viability_score'], reverse=True)
        
        return recommendations
    
    def get_all_materials(self):
        """Get all materials in the database"""
        return self.materials
    
    def get_material_properties(self, material_type):
        """Get properties for a specific material"""
        return self.materials.get(material_type, {}).get('properties', {})
    
    def get_recycling_methods(self, material_type):
        """Get recycling methods for a specific material"""
        return self.materials.get(material_type, {}).get('recycling_methods', [])
    
    def get_market_data(self, material_type):
        """Get market data for a specific material"""
        return self.materials.get(material_type, {}).get('market_data', {})
    
    def get_material_counts(self):
        """Get count of recommendation options per material"""
        counts = {}
        for material, data in self.materials.items():
            counts[material] = len(data.get('recommendations', []))
        return counts
    
    def get_recycling_difficulty(self):
        """Get recycling difficulty scores for visualization"""
        difficulty_scores = {
            'Plastic': 6.5,
            'Metal': 3.0,
            'Paper': 4.5,
            'Glass': 2.5,
            'Electronics': 8.5,
            'Textile': 7.0
        }
        return difficulty_scores
    
    def search_materials(self, query):
        """Search materials based on properties or methods"""
        results = []
        query_lower = query.lower()
        
        for material_type, data in self.materials.items():
            # Search in properties
            for prop, value in data.get('properties', {}).items():
                if query_lower in prop.lower() or query_lower in str(value).lower():
                    results.append({
                        'material': material_type,
                        'match_type': 'property',
                        'match_field': prop,
                        'match_value': value
                    })
            
            # Search in recycling methods
            for method in data.get('recycling_methods', []):
                if query_lower in method.lower():
                    results.append({
                        'material': material_type,
                        'match_type': 'method',
                        'match_field': 'recycling_method',
                        'match_value': method
                    })
        
        return results
