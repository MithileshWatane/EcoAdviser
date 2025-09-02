import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Import custom modules
from models.material_classifier import MaterialClassifier
from data.material_database import MaterialDatabase
from utils.carbon_calculator import CarbonCalculator
from utils.economic_analyzer import EconomicAnalyzer
from utils.image_processor import ImageProcessor

# Page configuration
st.set_page_config(
    page_title="AI Circular Economy Advisor",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components with caching"""
    classifier = MaterialClassifier()
    database = MaterialDatabase()
    carbon_calc = CarbonCalculator()
    economic_analyzer = EconomicAnalyzer()
    image_processor = ImageProcessor()
    
    return classifier, database, carbon_calc, economic_analyzer, image_processor

def main():
    # Header
    st.title("🌱 AI-Powered Circular Economy Advisor")
    st.markdown("*Intelligent Material Reuse & Waste Minimization*")
    
    # Initialize components
    classifier, database, carbon_calc, economic_analyzer, image_processor = initialize_components()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section:", [
        "📊 Material Analysis",
        "🔍 Waste Classification", 
        "♻️ Recommendation Engine",
        "📈 Impact Calculator",
        "🏢 Material Database"
    ])
    
    if page == "📊 Material Analysis":
        material_analysis_page(classifier, database, carbon_calc, economic_analyzer, image_processor)
    elif page == "🔍 Waste Classification":
        waste_classification_page(classifier, image_processor)
    elif page == "♻️ Recommendation Engine":
        recommendation_engine_page(classifier, database, carbon_calc, economic_analyzer)
    elif page == "📈 Impact Calculator":
        impact_calculator_page(carbon_calc, economic_analyzer)
    elif page == "🏢 Material Database":
        material_database_page(database)

def material_analysis_page(classifier, database, carbon_calc, economic_analyzer, image_processor):
    st.header("📊 Complete Material Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Material Information")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Text Description", "Image Upload"])
        
        if input_method == "Text Description":
            material_description = st.text_area(
                "Describe your waste material:",
                placeholder="e.g., Clear plastic bottles, aluminum cans, cardboard boxes..."
            )
            material_type = st.selectbox(
                "Material Type (if known):",
                ["Auto-detect", "Plastic", "Metal", "Paper", "Glass", "Electronics", "Textile"]
            )
            uploaded_file = None
        else:
            uploaded_file = st.file_uploader(
                "Upload material image:",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the waste material"
            )
            material_description = ""
            material_type = "Auto-detect"
        
        quantity = st.number_input("Quantity (kg):", min_value=0.1, value=10.0, step=0.1)
        contamination = st.slider("Contamination Level (%):", 0, 100, 20)
        
        analyze_button = st.button("🔍 Analyze Material", type="primary")
    
    with col2:
        st.subheader("Analysis Results")
        
        if analyze_button:
            # Process input
            if input_method == "Image Upload" and uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Material", use_container_width=True)
                detected_material = image_processor.classify_image(image)
                material_description = f"Detected: {detected_material}"
            
            if material_description or (input_method == "Image Upload" and uploaded_file):
                # Classification
                if material_type == "Auto-detect":
                    classified_material = classifier.classify_material(material_description)
                else:
                    classified_material = material_type.lower()
                
                st.success(f"**Classified Material:** {classified_material.title()}")
                
                # Get recommendations
                recommendations = database.get_recommendations(classified_material, contamination)
                
                # Display recommendations
                for i, rec in enumerate(recommendations[:3]):
                    with st.expander(f"Option {i+1}: {rec['action'].title()} ({rec['viability_score']:.1f}/10)"):
                        st.write(f"**Method:** {rec['method']}")
                        st.write(f"**Description:** {rec['description']}")
                        
                        # Carbon impact
                        carbon_saved = carbon_calc.calculate_savings(classified_material, rec['action'], quantity)
                        st.metric("CO₂ Saved", f"{carbon_saved:.1f} kg", delta=f"{carbon_saved/quantity:.2f} kg/kg material")
                        
                        # Economic impact
                        economic_value = economic_analyzer.calculate_value(classified_material, rec['action'], quantity, contamination)
                        st.metric("Economic Value", f"${economic_value:.2f}", delta=f"${economic_value/quantity:.2f}/kg")
            else:
                st.info("Please provide material information to begin analysis.")

def waste_classification_page(classifier, image_processor):
    st.header("🔍 Waste Material Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Classification Input")
        
        # Batch classification option
        classification_type = st.radio("Classification Type:", ["Single Item", "Batch Processing"])
        
        if classification_type == "Single Item":
            description = st.text_area(
                "Material Description:",
                placeholder="Describe the material properties, appearance, and any identifying features..."
            )
            
            if st.button("Classify Material"):
                if description:
                    result = classifier.classify_material(description)
                    confidence = classifier.get_confidence_score(description)
                    
                    st.success(f"**Material Type:** {result.title()}")
                    
                    # Show classification details
                    details = classifier.get_classification_details(result)
                    st.json(details)
        else:
            st.subheader("Batch Processing")
            batch_descriptions = st.text_area(
                "Enter multiple materials (one per line):",
                height=150,
                placeholder="Clear plastic bottle\nAluminum can\nCardboard box\n..."
            )
            
            if st.button("Classify Batch"):
                if batch_descriptions:
                    materials = [desc.strip() for desc in batch_descriptions.split('\n') if desc.strip()]
                    results = []
                    
                    for material in materials:
                        classification = classifier.classify_material(material)
                        confidence = classifier.get_confidence_score(material)
                        results.append({
                            'Material Description': material,
                            'Classification': classification.title(),
                            'Confidence': f"{confidence:.1%}"
                        })
                    
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("Classification Guide")
        
        # Material type distribution
        sample_data = classifier.get_sample_classifications()
        fig = px.pie(
            values=list(sample_data.values()),
            names=list(sample_data.keys()),
            title="Common Material Types"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification tips
        st.markdown("### 💡 Classification Tips")
        st.markdown("""
        **For better accuracy:**
        - Include color, texture, and size information
        - Mention any labels or markings
        - Describe the original use or source
        - Note the condition (new, used, damaged)
        """)

def recommendation_engine_page(classifier, database, carbon_calc, economic_analyzer):
    st.header("♻️ Waste Strategy Recommendation Engine")
    
    # Material selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        material_type = st.selectbox(
            "Material Type:",
            ["Plastic", "Metal", "Paper", "Glass", "Electronics", "Textile"]
        )
    
    with col2:
        quantity = st.number_input("Quantity (kg):", min_value=0.1, value=100.0)
    
    with col3:
        contamination = st.slider("Contamination %:", 0, 100, 15)
    
    # Get recommendations
    recommendations = database.get_recommendations(material_type.lower(), contamination)
    
    st.subheader("📋 Recommended Strategies")
    
    # Create comparison chart
    strategies = []
    carbon_savings = []
    economic_values = []
    viability_scores = []
    
    for rec in recommendations[:4]:  # Top 4 recommendations
        strategies.append(rec['action'].title())
        carbon_savings.append(carbon_calc.calculate_savings(material_type.lower(), rec['action'], quantity))
        economic_values.append(economic_analyzer.calculate_value(material_type.lower(), rec['action'], quantity, contamination))
        viability_scores.append(rec['viability_score'])
    
    # Multi-metric comparison
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Carbon Savings (kg CO₂)',
        x=strategies,
        y=carbon_savings,
        yaxis='y',
        offsetgroup=1
    ))
    
    fig.add_trace(go.Bar(
        name='Economic Value ($)',
        x=strategies,
        y=economic_values,
        yaxis='y2',
        offsetgroup=2
    ))
    
    fig.update_layout(
        title='Strategy Comparison: Environmental vs Economic Impact',
        xaxis_title='Strategies',
        yaxis=dict(title='Carbon Savings (kg CO₂)', side='left'),
        yaxis2=dict(title='Economic Value ($)', side='right', overlaying='y'),
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed recommendations
    for i, rec in enumerate(recommendations):
        with st.expander(f"Strategy {i+1}: {rec['action'].title()} - Viability Score: {rec['viability_score']:.2f}/10"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Method:** {rec['method']}")
                st.markdown(f"**Process:** {rec['description']}")
                st.markdown(f"**Requirements:** {rec.get('requirements', 'Standard processing facilities')}")
                
                # Market potential
                if 'market_demand' in rec:
                    st.markdown(f"**Market Demand:** {rec['market_demand']}")
            
            with col2:
                # Impact metrics
                carbon_saved = carbon_calc.calculate_savings(material_type.lower(), rec['action'], quantity)
                economic_value = economic_analyzer.calculate_value(material_type.lower(), rec['action'], quantity, contamination)
                
                st.metric("CO₂ Savings", f"{carbon_saved:.1f} kg")
                st.metric("Economic Value", f"${economic_value:.2f}")
                st.metric("Processing Cost", f"${economic_value * 0.3:.2f}")

def impact_calculator_page(carbon_calc, economic_analyzer):
    st.header("📈 Environmental & Economic Impact Calculator")
    
    st.subheader("⚙️ Calculator Settings")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        material_type = st.selectbox("Material:", ["Plastic", "Metal", "Paper", "Glass", "Electronics", "Textile"])
    
    with col2:
        strategy = st.selectbox("Strategy:", ["Reuse", "Recycle", "Upcycle", "Dispose"])
    
    with col3:
        quantity = st.number_input("Quantity (kg):", min_value=0.1, value=500.0)
    
    with col4:
        time_period = st.selectbox("Time Period:", ["Monthly", "Quarterly", "Annually"])
    
    # Multiplier based on time period
    multipliers = {"Monthly": 1, "Quarterly": 3, "Annually": 12}
    total_quantity = quantity * multipliers[time_period]
    
    if st.button("Calculate Impact", type="primary"):
        st.subheader("🌍 Environmental Impact")
        
        carbon_saved = carbon_calc.calculate_savings(material_type.lower(), strategy.lower(), total_quantity)
        carbon_baseline = carbon_calc.get_disposal_impact(material_type.lower(), total_quantity)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CO₂ Saved", f"{carbon_saved:.0f} kg", delta=f"vs disposal")
        
        with col2:
            st.metric("Equivalent Trees", f"{carbon_saved/22:.0f}", delta="trees planted")
        
        with col3:
            st.metric("Car Miles Offset", f"{carbon_saved*2.3:.0f}", delta="miles")
        
        # Economic impact
        st.subheader("💰 Economic Impact")
        
        economic_value = economic_analyzer.calculate_value(material_type.lower(), strategy.lower(), total_quantity, 20)
        processing_cost = economic_value * 0.35
        net_benefit = economic_value - processing_cost
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gross Value", f"${economic_value:.2f}")
        
        with col2:
            st.metric("Processing Cost", f"${processing_cost:.2f}")
        
        with col3:
            st.metric("Net Benefit", f"${net_benefit:.2f}", delta=f"${net_benefit/total_quantity:.2f}/kg")
        
        # Impact visualization
        impact_data = {
            'Metric': ['Carbon Saved (kg)', 'Economic Value ($)', 'Processing Cost ($)', 'Net Benefit ($)'],
            'Value': [carbon_saved, economic_value, processing_cost, net_benefit]
        }
        
        fig = px.bar(impact_data, x='Metric', y='Value', title=f'Impact Summary - {time_period} Projection')
        st.plotly_chart(fig, use_container_width=True)

def material_database_page(database):
    st.header("🏢 Material Database & Properties")
    
    # Database overview
    materials = database.get_all_materials()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Database Overview")
        
        material_counts = database.get_material_counts()
        for material, count in material_counts.items():
            st.metric(f"{material.title()} Records", count)
    
    with col2:
        st.subheader("Material Properties Comparison")
        
        # Recycling difficulty comparison
        difficulty_data = database.get_recycling_difficulty()
        fig = px.bar(
            x=list(difficulty_data.keys()),
            y=list(difficulty_data.values()),
            title="Recycling Difficulty by Material Type",
            labels={'x': 'Material Type', 'y': 'Difficulty Score (1-10)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed material information
    st.subheader("📋 Detailed Material Information")
    
    selected_material = st.selectbox("Select Material for Details:", list(materials.keys()))
    
    if selected_material:
        material_info = materials[selected_material]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Properties")
            for key, value in material_info['properties'].items():
                st.write(f"**{key.title()}:** {value}")
        
        with col2:
            st.markdown("### Recycling Methods")
            for method in material_info['recycling_methods']:
                st.write(f"• {method}")
        
        # Market data
        if 'market_data' in material_info:
            st.subheader("Market Information")
            market_data = material_info['market_data']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${market_data['price_per_kg']:.2f}/kg")
            
            with col2:
                st.metric("Market Demand", market_data['demand_level'])
            
            with col3:
                st.metric("Price Trend", market_data['trend'])

if __name__ == "__main__":
    main()
