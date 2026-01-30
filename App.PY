"""
Fibonacci Retracement Analysis - Streamlit Web Application
Based on Tsinaslanidis (2022) methodology
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from io import BytesIO
import base64

# Import the core analysis module
from fibonacci_zones_improved import TsinaslanidisFibonacciZones

# Page configuration
st.set_page_config(
    page_title="Fibonacci Retracement Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def create_price_chart(analyzer):
    """Create interactive price chart with Fibonacci zones."""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=analyzer.data.index,
        y=analyzer.data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='black', width=2),
        hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))
    
    # Add Fibonacci zones
    colors = ['rgba(31, 119, 180, 0.2)', 'rgba(255, 127, 14, 0.2)', 
              'rgba(44, 160, 44, 0.2)', 'rgba(214, 39, 40, 0.2)', 
              'rgba(148, 103, 189, 0.2)']
    
    for idx, (label, zone) in enumerate(analyzer.fib_zones.items()):
        color = colors[idx % len(colors)]
        
        # Add zone as filled area
        fig.add_trace(go.Scatter(
            x=[analyzer.data.index[0], analyzer.data.index[-1], 
               analyzer.data.index[-1], analyzer.data.index[0]],
            y=[zone['zone_low'], zone['zone_low'], 
               zone['zone_high'], zone['zone_high']],
            fill='toself',
            fillcolor=color,
            line=dict(width=0),
            name=f'Fib {label}',
            showlegend=True,
            hovertemplate=f'{label}<br>Range: {zone["zone_low"]:.2f} - {zone["zone_high"]:.2f}<extra></extra>'
        ))
        
        # Add level line
        fig.add_hline(
            y=zone['level'],
            line_dash="dash",
            line_color=color.replace('0.2', '0.8'),
            annotation_text=label,
            annotation_position="right"
        )
    
    # Add swing points
    fig.add_trace(go.Scatter(
        x=[analyzer.swing_start_date],
        y=[analyzer.swing_start],
        mode='markers',
        name='Swing Start',
        marker=dict(color='green', size=15, symbol='circle'),
        hovertemplate='Start: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[analyzer.swing_end_date],
        y=[analyzer.swing_end],
        mode='markers',
        name='Swing End',
        marker=dict(color='red', size=15, symbol='circle'),
        hovertemplate='End: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{analyzer.symbol} - Fibonacci Retracement Zones',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )
    
    return fig

def create_bounce_comparison_chart(analyzer, period=1):
    """Create bar chart comparing bounce probabilities."""
    fib_labels = list(analyzer.fib_zones.keys())
    fib_probs = [analyzer.bounce_stats['fibonacci']['probabilities'][label][period] * 100 
                 for label in fib_labels]
    
    non_fib_labels = list(analyzer.non_fib_zones.keys())
    non_fib_probs = [analyzer.bounce_stats['non_fibonacci']['probabilities'][label][period] * 100 
                     for label in non_fib_labels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=fib_labels,
        y=fib_probs,
        name='Fibonacci Zones',
        marker_color='rgb(31, 119, 180)',
        hovertemplate='%{x}<br>Bounce Probability: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=non_fib_labels[:len(fib_labels)],
        y=non_fib_probs[:len(fib_labels)],
        name='Random Zones',
        marker_color='rgb(214, 39, 40)',
        hovertemplate='%{x}<br>Bounce Probability: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Bounce Probability Comparison ({period}-day lookforward)',
        xaxis_title='Zone',
        yaxis_title='Bounce Probability (%)',
        barmode='group',
        height=400,
        hovermode='x'
    )
    
    return fig

def create_zone_width_chart(width_results):
    """Create line chart for zone width analysis."""
    widths = [r['zone_width_percent'] * 100 for r in width_results]
    fib_probs = [r['bounce_prob_fib'] * 100 for r in width_results]
    non_fib_probs = [r['bounce_prob_non_fib'] * 100 for r in width_results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=widths,
        y=fib_probs,
        mode='lines+markers',
        name='Fibonacci Zones',
        line=dict(color='rgb(31, 119, 180)', width=3),
        marker=dict(size=10),
        hovertemplate='Width: %{x:.2f}%<br>Bounce Prob: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=widths,
        y=non_fib_probs,
        mode='lines+markers',
        name='Random Zones',
        line=dict(color='rgb(214, 39, 40)', width=3, dash='dash'),
        marker=dict(size=10),
        hovertemplate='Width: %{x:.2f}%<br>Bounce Prob: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Zone Width vs Bounce Probability',
        xaxis_title='Zone Width (%)',
        yaxis_title='Bounce Probability (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def get_download_link(df, filename, text):
    """Generate download link for dataframe."""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">📊 Fibonacci Retracement Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Based on Tsinaslanidis (2022) Methodology</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Input parameters
        st.subheader("1️⃣ Stock Selection")
        symbol = st.text_input(
            "Stock Symbol",
            value="^NSEI",
            help="Enter ticker symbol (e.g., ^NSEI for NIFTY, AAPL for Apple)"
        )
        
        st.subheader("2️⃣ Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=3*365),
                help="Analysis start date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Analysis end date"
            )
        
        st.subheader("3️⃣ Analysis Parameters")
        
        zeta = st.slider(
            "Zone Width (ζ)",
            min_value=0.005,
            max_value=0.030,
            value=0.010,
            step=0.001,
            format="%.3f",
            help="Percentage for zone construction (e.g., 0.01 = 1%)"
        )
        
        window = st.slider(
            "Swing Detection Window",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Window size for identifying swing points"
        )
        
        prominence = st.slider(
            "Prominence Threshold",
            min_value=0.01,
            max_value=0.05,
            value=0.02,
            step=0.001,
            format="%.3f",
            help="Minimum significance for swing points"
        )
        
        use_extended = st.checkbox(
            "Use Extended Fibonacci Ratios",
            value=False,
            help="Include 50% and 78.6% levels"
        )
        
        st.subheader("4️⃣ Advanced Options")
        
        lookforward = st.multiselect(
            "Lookforward Periods (days)",
            options=[1, 2, 3, 5, 7, 10],
            default=[1, 3, 5],
            help="Number of days to check for bounce confirmation"
        )
        
        bounce_threshold = st.slider(
            "Bounce Threshold",
            min_value=0.001,
            max_value=0.010,
            value=0.003,
            step=0.001,
            format="%.3f",
            help="Minimum price movement to qualify as bounce"
        )
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.info("💡 Tip: Start with default parameters and adjust based on results")
    
    # Main content area
    if run_analysis:
        if not symbol:
            st.error("Please enter a stock symbol")
            return
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch data
            status_text.text("📥 Fetching data...")
            progress_bar.progress(10)
            
            analyzer = TsinaslanidisFibonacciZones(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            analyzer.fetch_data()
            
            # Step 2: Identify swings
            status_text.text("🔍 Identifying swing points...")
            progress_bar.progress(25)
            analyzer.identify_swing_points(window=window, prominence_threshold=prominence)
            
            # Step 3: Identify trend
            status_text.text("📈 Identifying major trend...")
            progress_bar.progress(40)
            analyzer.identify_major_trend(lookback_days=100)
            
            # Step 4: Calculate Fibonacci levels
            status_text.text("📊 Calculating Fibonacci levels...")
            progress_bar.progress(55)
            analyzer.calculate_fibonacci_levels(use_extended=use_extended)
            
            # Step 5: Construct zones
            status_text.text("🎯 Constructing zones...")
            progress_bar.progress(70)
            analyzer.construct_zones(zeta=zeta)
            
            # Step 6: Detect bounces
            status_text.text("🔄 Detecting bounces...")
            progress_bar.progress(85)
            analyzer.detect_bounces(
                lookforward_periods=lookforward,
                bounce_threshold=bounce_threshold
            )
            
            # Step 7: Statistical analysis
            status_text.text("📉 Running statistical tests...")
            progress_bar.progress(95)
            t_stat, p_value, u_pvalue = analyzer.statistical_comparison(period=lookforward[0])
            analyzer.t_stat = t_stat
            analyzer.p_value = p_value
            
            # Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            # Display results
            st.success("Analysis completed successfully!")
            
            # Results tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", 
                "📈 Price Chart", 
                "📉 Statistics", 
                "🔬 Zone Analysis",
                "📄 Report"
            ])
            
            with tab1:
                st.header("Analysis Overview")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Trend Direction",
                        analyzer.trend_direction.upper(),
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Price Range",
                        f"{analyzer.price_range:.2f}",
                        delta=f"{(analyzer.price_range/analyzer.swing_start)*100:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "Trading Days",
                        len(analyzer.data),
                        delta=None
                    )
                
                with col4:
                    period = lookforward[0]
                    fib_prob = analyzer.bounce_stats['fibonacci']['avg_probabilities'][period]
                    st.metric(
                        "Avg Fib Bounce %",
                        f"{fib_prob:.2%}",
                        delta=None
                    )
                
                # Swing information
                st.subheader("Trend Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **Swing Start**  
                    Date: {analyzer.swing_start_date.strftime('%Y-%m-%d')}  
                    Price: {analyzer.swing_start:.2f}
                    """)
                
                with col2:
                    st.info(f"""
                    **Swing End**  
                    Date: {analyzer.swing_end_date.strftime('%Y-%m-%d')}  
                    Price: {analyzer.swing_end:.2f}
                    """)
                
                # Fibonacci levels table
                st.subheader("Fibonacci Retracement Levels")
                fib_df = pd.DataFrame([
                    {
                        'Level': label,
                        'Price': f"{zone['level']:.2f}",
                        'Zone Range': f"{zone['zone_low']:.2f} - {zone['zone_high']:.2f}",
                        'Width %': f"{zone['width_pct']:.2f}%"
                    }
                    for label, zone in analyzer.fib_zones.items()
                ])
                st.dataframe(fib_df, use_container_width=True)
            
            with tab2:
                st.header("Interactive Price Chart")
                fig_price = create_price_chart(analyzer)
                st.plotly_chart(fig_price, use_container_width=True)
                
                st.info("""
                **Chart Guide:**
                - 🟢 Green marker: Swing start point
                - 🔴 Red marker: Swing end point  
                - Colored bands: Fibonacci retracement zones
                - Dashed lines: Exact Fibonacci levels
                """)
            
            with tab3:
                st.header("Statistical Analysis")
                
                # Hypothesis test results
                st.subheader("🧪 Hypothesis Testing")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "T-test p-value",
                        f"{p_value:.4f}",
                        delta="Significant" if p_value < 0.05 else "Not Significant"
                    )
                
                with col2:
                    st.metric(
                        "Mann-Whitney U p-value",
                        f"{u_pvalue:.4f}",
                        delta="Significant" if u_pvalue < 0.05 else "Not Significant"
                    )
                
                if p_value >= 0.05:
                    st.success("""
                    ✅ **No significant difference found** (p ≥ 0.05)
                    
                    This is consistent with Tsinaslanidis (2022) findings, suggesting that 
                    Fibonacci retracement levels do not provide superior predictive power 
                    compared to random price levels.
                    """)
                else:
                    st.warning("""
                    ⚠️ **Significant difference detected** (p < 0.05)
                    
                    This differs from Tsinaslanidis (2022) findings. The significance may be 
                    due to specific market conditions, sample size, or parameter choices.
                    """)
                
                # Bounce probability comparison
                st.subheader("📊 Bounce Probability Comparison")
                period = lookforward[0]
                fig_bounce = create_bounce_comparison_chart(analyzer, period)
                st.plotly_chart(fig_bounce, use_container_width=True)
                
                # Detailed statistics table
                st.subheader("📋 Detailed Statistics")
                
                stats_data = []
                for label in analyzer.fib_zones.keys():
                    stats_data.append({
                        'Zone': f'Fib {label}',
                        'Type': 'Fibonacci',
                        'Hits': analyzer.bounce_stats['fibonacci']['hits'][label],
                        'Bounces': analyzer.bounce_stats['fibonacci']['bounces'][label][period],
                        'Probability': f"{analyzer.bounce_stats['fibonacci']['probabilities'][label][period]:.2%}"
                    })
                
                for label in analyzer.non_fib_zones.keys():
                    stats_data.append({
                        'Zone': label,
                        'Type': 'Random',
                        'Hits': analyzer.bounce_stats['non_fibonacci']['hits'][label],
                        'Bounces': analyzer.bounce_stats['non_fibonacci']['bounces'][label][period],
                        'Probability': f"{analyzer.bounce_stats['non_fibonacci']['probabilities'][label][period]:.2%}"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
                # Download link
                st.markdown(
                    get_download_link(stats_df, f"{symbol}_statistics.csv", "📥 Download Statistics CSV"),
                    unsafe_allow_html=True
                )
            
            with tab4:
                st.header("Zone Width Analysis")
                
                with st.spinner("Running zone width analysis..."):
                    width_results = analyzer.analyze_zone_width_relationship(
                        zeta_values=[0.005, 0.01, 0.015, 0.02, 0.025]
                    )
                    analyzer.width_analysis_results = width_results
                
                # Restore original zones
                analyzer.construct_zones(zeta=zeta)
                analyzer.detect_bounces(lookforward_periods=lookforward, bounce_threshold=bounce_threshold)
                
                # Chart
                fig_width = create_zone_width_chart(width_results)
                st.plotly_chart(fig_width, use_container_width=True)
                
                # Results table
                st.subheader("Zone Width Results")
                width_df = pd.DataFrame([
                    {
                        'ζ': r['zeta'],
                        'Zone Width %': f"{r['zone_width_percent']:.2%}",
                        'Fib Bounce Prob': f"{r['bounce_prob_fib']:.2%}",
                        'Random Bounce Prob': f"{r['bounce_prob_non_fib']:.2%}",
                        'Fib Hits': r['total_hits_fib'],
                        'Random Hits': r['total_hits_non_fib']
                    }
                    for r in width_results
                ])
                st.dataframe(width_df, use_container_width=True)
                
                # Correlation analysis
                widths = [r['zone_width_percent'] for r in width_results]
                fib_probs = [r['bounce_prob_fib'] for r in width_results]
                
                correlation = np.corrcoef(widths, fib_probs)[0, 1]
                
                st.metric(
                    "Correlation (Width vs Fib Bounce Prob)",
                    f"{correlation:.4f}",
                    delta="Positive" if correlation > 0 else "Negative"
                )
                
                if correlation > 0:
                    st.info("📈 Positive correlation found: Wider zones tend to have higher bounce probabilities")
                else:
                    st.info("📉 Negative correlation found: Wider zones tend to have lower bounce probabilities")
            
            with tab5:
                st.header("Analysis Report")
                
                # Generate and display report
                report = analyzer.generate_report()
                st.text(report)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"{symbol}_fibonacci_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
                
                # Citation
                st.subheader("📚 Citation")
                st.code(f"""
# BibTeX Citation
@software{{fibonacci_analysis_{datetime.now().year},
  title = {{Fibonacci Retracement Analysis for {symbol}}},
  author = {{Your Name}},
  year = {{{datetime.now().year}}},
  note = {{Analysis performed on {datetime.now().strftime('%Y-%m-%d')}}}
}}

# Reference
Tsinaslanidis, P. E. (2022). 
"Fibonacci retracements: An empirical investigation"
                """, language="bibtex")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    else:
        # Welcome screen
        st.info("""
        👈 **Configure your analysis in the sidebar and click "Run Analysis"**
        
        This tool implements the methodology from Tsinaslanidis (2022) to empirically 
        test whether Fibonacci retracement levels provide superior predictive power 
        compared to random price levels.
        """)
        
        with st.expander("📖 About This Tool"):
            st.markdown("""
            ### Research Background
            
            This application implements the empirical methodology described in:
            
            > Tsinaslanidis, P. E. (2022). "Fibonacci retracements: An empirical investigation"
            
            ### What Does It Do?
            
            1. **Identifies significant price swings** in historical data
            2. **Calculates Fibonacci retracement levels** (23.6%, 38.2%, 61.8%, etc.)
            3. **Constructs zones** around each level
            4. **Analyzes price bounces** at these zones
            5. **Compares Fibonacci zones** with random zones statistically
            
            ### Key Findings (Tsinaslanidis 2022)
            
            - Fibonacci levels do **not** show superior predictive power vs random levels
            - Zone width is **positively related** to bounce probability
            - This relationship holds for **both** Fibonacci and random zones
            - Suggests any predictive power may be **self-fulfilling**
            
            ### How to Use
            
            1. Enter a stock symbol (e.g., `^NSEI`, `AAPL`, `RELIANCE.NS`)
            2. Select date range (recommended: 2-5 years)
            3. Adjust parameters if needed (defaults work well)
            4. Click "Run Analysis"
            5. Explore results in different tabs
            """)
        
        with st.expander("🎯 Sample Symbols"):
            st.markdown("""
            **Indian Markets:**
            - `^NSEI` - NIFTY 50
            - `^NSEBANK` - NIFTY Bank
            - `RELIANCE.NS` - Reliance Industries
            - `TCS.NS` - Tata Consultancy Services
            
            **US Markets:**
            - `^GSPC` - S&P 500
            - `^DJI` - Dow Jones
            - `AAPL` - Apple Inc.
            - `MSFT` - Microsoft
            - `TSLA` - Tesla
            
            **Crypto (if available):**
            - `BTC-USD` - Bitcoin
            - `ETH-USD` - Ethereum
            """)

if __name__ == "__main__":
    main()
