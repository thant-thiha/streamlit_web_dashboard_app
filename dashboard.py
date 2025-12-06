import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page for optimal senior user experience
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛒",
    layout="wide",  # Wide layout reduces scrolling
    initial_sidebar_state="collapsed"  # Simpler single-page view
)

# Custom CSS for Senior-Friendly Design - Larger fonts, high contrast, clear spacing
st.markdown("""
    <style>
    /* Increase base font size for better readability */
    .stApp {
        font-size: 16px;
    }
    
    /* Large, prominent headers */
    h1 {
        font-size: 42px !important;
        color: #1f4788;
        font-weight: bold;
    }
    
    h2 {
        font-size: 32px !important;
        color: #2c5aa0;
        margin-top: 30px;
    }
    
    h3 {
        font-size: 24px !important;
        color: #3d6bb3;
    }
    
    /* Large metric displays */
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
    }
    
    /* High contrast buttons */
    .stButton button {
        font-size: 18px;
        padding: 15px 30px;
        background-color: #1f4788;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background-color: #2c5aa0;
    }
    
    /* Clear spacing between sections */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Accessible select boxes */
    .stSelectbox label {
        font-size: 18px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load datasets
product = pd.read_csv('data/product.csv')
hh_demographic = pd.read_csv('data/hh_demographic.csv')
campaign_table = pd.read_csv('data/campaign_table.csv')
campaign_desc = pd.read_csv('data/campaign_desc.csv')

# Load compressed CSV
transaction_data = pd.read_csv('data/transaction_data.csv.gz', compression='gzip')

# Define the start date (Day 1)
start_date = pd.to_datetime('2023-01-01')

# Calculate the new 'DATE' column:
# 1. Access the 'DAY' column in your DataFrame.
# 2. Subtract 1 (since Day 1 has a zero-day offset).
# 3. Use pd.to_timedelta to convert the day difference into a time difference.
# 4. Add the timedelta to the start_date.

transaction_data['DATE'] = start_date + pd.to_timedelta(
    transaction_data['DAY'] - 1, 
    unit='D'
)

# Convert 'START_DAY' to 'START_DATE' in campaign table
campaign_desc['START_DATE'] = start_date + pd.to_timedelta(
    campaign_desc['START_DAY'] - 1, 
    unit='D'
)

# Convert 'END_DAY' to 'END_DATE' in campaign table
campaign_desc['END_DATE'] = start_date + pd.to_timedelta(
    campaign_desc['END_DAY'] - 1, 
    unit='D'
)

# Add product information to transactions
df = transaction_data.merge(
    product[['PRODUCT_ID', 'DEPARTMENT', 'BRAND', 'COMMODITY_DESC']], 
    left_on='PRODUCT_ID',
    right_on='PRODUCT_ID',
    how='left'
)

# Add demographic information (select key classification variables)
df = df.merge(
    hh_demographic[['household_key', 'classification_1', 'classification_2', 
                    'classification_3', 'classification_5']], 
    on='household_key', 
    how='left'
)

# Rename for clarity
df = df.rename(columns={
    'classification_1': 'DEMOGRAPHIC_GROUP',      # Group1 through Group6
    'classification_2': 'DEMOGRAPHIC_TYPE',        # X, Y, Z
    'classification_3': 'DEMOGRAPHIC_LEVEL',       # Level1 through Level12
    'classification_5': 'SHOPPING_SEGMENT'         # Group1 through Group6
})

# Add campaign participation
campaign_participation = campaign_table[['household_key', 'CAMPAIGN', 'DESCRIPTION']].copy()
campaign_participation['IN_CAMPAIGN'] = 1

# Get first campaign per household (if multiple campaigns)
campaign_participation = campaign_participation.groupby('household_key').first().reset_index()

df = df.merge(
    campaign_participation, 
    on='household_key', 
    how='left'
)

df['IN_CAMPAIGN'] = df['IN_CAMPAIGN'].fillna(0).astype(int)
df['CAMPAIGN_TYPE'] = df['DESCRIPTION'].fillna('No Campaign')

# Temporal features (critical for time-series forecasting)
df['MONTH'] = df['DATE'].dt.month
df['MONTH_NAME'] = df['DATE'].dt.strftime('%B')
df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
df['DAY_NAME'] = df['DATE'].dt.strftime('%A')
df['QUARTER'] = df['DATE'].dt.quarter
df['YEAR'] = df['DATE'].dt.year
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)

# Discount features (for price optimization ML)
df['TOTAL_DISCOUNT'] = (
    df['COUPON_MATCH_DISC'] + 
    df['COUPON_DISC'] + 
    df['RETAIL_DISC']
)
df['DISCOUNT_RATE'] = (
    df['TOTAL_DISCOUNT'] / 
    (df['SALES_VALUE'] + df['TOTAL_DISCOUNT'])
).fillna(0)

# Revenue features
df['NET_REVENUE'] = df['SALES_VALUE']  # Already net of discounts
df['UNIT_PRICE'] = df['SALES_VALUE'] / df['QUANTITY']
df['HAS_DISCOUNT'] = (df['TOTAL_DISCOUNT'] > 0).astype(int)

# Customer-Level Aggregations
# Customer lifetime value and segmentation features for ML
customer_metrics = df.groupby('household_key').agg({
    'BASKET_ID': 'nunique',          # Number of shopping trips
    'SALES_VALUE': 'sum',             # Total spent
    'QUANTITY': 'sum',                # Total items bought
    'DATE': ['min', 'max'],            # First and last purchase
    'TOTAL_DISCOUNT': 'sum',          # Total discounts received
    'STORE_ID': 'nunique'             # Number of different stores visited
}).reset_index()

customer_metrics.columns = ['household_key', 'NUM_TRIPS', 'TOTAL_SPENT', 
                            'TOTAL_ITEMS', 'FIRST_PURCHASE', 'LAST_PURCHASE',
                            'TOTAL_DISCOUNTS', 'NUM_STORES']

customer_metrics['DAYS_ACTIVE'] = (
    customer_metrics['LAST_PURCHASE'] - customer_metrics['FIRST_PURCHASE']
).dt.days + 1

customer_metrics['AVG_BASKET_VALUE'] = (
    customer_metrics['TOTAL_SPENT'] / customer_metrics['NUM_TRIPS']
)

customer_metrics['ITEMS_PER_TRIP'] = (
    customer_metrics['TOTAL_ITEMS'] / customer_metrics['NUM_TRIPS']
)

customer_metrics['DISCOUNT_RATE'] = (
    customer_metrics['TOTAL_DISCOUNTS'] / 
    (customer_metrics['TOTAL_SPENT'] + customer_metrics['TOTAL_DISCOUNTS'])
)

product_performance = df.groupby('PRODUCT_ID').agg({
    'QUANTITY': 'sum',
    'SALES_VALUE': 'sum',
    'BASKET_ID': 'nunique',
    'household_key': 'nunique',
    'TOTAL_DISCOUNT': 'sum'
}).reset_index()

product_performance.columns = ['PRODUCT_ID', 'TOTAL_QUANTITY', 'TOTAL_SALES', 
                               'NUM_BASKETS', 'NUM_CUSTOMERS', 'TOTAL_DISCOUNTS']

# Merge back product details
product_performance = product_performance.merge(
    product[['PRODUCT_ID', 'DEPARTMENT', 'BRAND', 'COMMODITY_DESC']], 
    left_on='PRODUCT_ID',
    right_on='PRODUCT_ID',
    how='left'
)

product_performance['AVG_PRICE'] = (
    product_performance['TOTAL_SALES'] / product_performance['TOTAL_QUANTITY']
)

dept_performance = df.groupby('DEPARTMENT').agg({
    'SALES_VALUE': 'sum',
    'QUANTITY': 'sum',
    'BASKET_ID': 'nunique',
    'household_key': 'nunique'
}).reset_index()

dept_performance.columns = ['DEPARTMENT', 'TOTAL_REVENUE', 'TOTAL_QUANTITY',
                            'NUM_BASKETS', 'NUM_CUSTOMERS']

campaign_metrics = df.groupby(['household_key', 'IN_CAMPAIGN']).agg({
    'SALES_VALUE': 'sum',
    'BASKET_ID': 'nunique',
    'QUANTITY': 'sum'
}).reset_index()

# HEADER SECTION
st.title("Retail Business Intelligence Dashboard")
st.markdown("""
<div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
    <h3 style='color: #1f4788; margin-top: 0;'>
        2-Year Analysis of 2,500 Frequent Shopper Households in a retail shop
    </h3>
    <p style='font-size: 18px; margin-bottom: 0;'>
        This dashboard analyzes purchasing patterns from our most valuable customers over 
        a 2-year period. All charts are interactive - simply hover over them to see detailed information.
    </p>
</div>
""", unsafe_allow_html=True)

# KEY METRICS SECTION
# Large numbers at top provide immediate context
st.header("Business Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = df['SALES_VALUE'].sum()
    st.metric(
        label="Total Revenue in 2 years",
        value=f"${total_revenue:,.0f}"
    )

with col2:
    avg_basket = df.groupby('BASKET_ID')['SALES_VALUE'].sum().mean()
    st.metric(
        label="Average Basket Size per Trip",
        value=f"${avg_basket:.2f}"
    )

with col3:
    unique_customers = df['household_key'].nunique()
    st.metric(
        label="Active Households/Frequent Shoppers",
        value=f"{unique_customers:,}"
    )

with col4:
    total_items = df['QUANTITY'].sum()
    st.metric(
        label="Items Sold in Units",
        value=f"{total_items:,.0f}"
    )


# SECTION 1: SALES TRENDS (Time-series for forecasting ML)

col1, col2 = st.columns(2)

# LEFT COLUMN: SALES TRENDS OVER TIME
with col1:
    st.subheader("Sales Trends Over Time")
    st.markdown("""
    <p style='font-size: 18px; color: #555;'>
        2 years of historical data captures seasonal patterns and behavior changes.
    </p>
    """, unsafe_allow_html=True)
    
    # Monthly sales aggregation
    monthly_sales = df.groupby(df['DATE'].dt.to_period('M')).agg({
        'SALES_VALUE': 'sum',
        'BASKET_ID': 'nunique',
        'QUANTITY': 'sum'
    }).reset_index()
    monthly_sales['DATE'] = monthly_sales['DATE'].dt.to_timestamp()
    
    # Create line chart
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_sales['DATE'],
        y=monthly_sales['SALES_VALUE'],
        mode='lines+markers',
        name='Monthly Revenue',
        line=dict(color='#1f4788', width=4),
        marker=dict(size=10),
        hovertemplate='<b>%{x|%B %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
    ))
    
    fig_trend.update_layout(
        height=400,
        font=dict(size=14),
        xaxis=dict(
            title=dict(
                text='Month',
                font=dict(size=16)),
            tickfont=dict(size=12),
            gridcolor='#e0e0e0'
        ),
        yaxis=dict(
            title=dict(
                text='Revenue ($)',
                font=dict(size=16)),
            tickfont=dict(size=12),
            gridcolor='#e0e0e0'
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        margin=dict(l=60, r=20, t=20, b=60)
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Year-over-year comparison
    yearly_sales = df.groupby('YEAR')['SALES_VALUE'].sum().reset_index()
    if len(yearly_sales) >= 2:
        subcol1, subcol2 = st.columns(2)
        for idx, row in yearly_sales.iterrows():
            with subcol1 if idx == 0 else subcol2:
                year_label = "Year 2023 Revenue" if idx == 0 else "Year 2024 Revenue"
                st.metric(year_label, f"${row['SALES_VALUE']:,.0f}")
    
    # ML explanation
    with st.expander("Click here to see how this data helps Machine Learning", expanded=False):
        st.markdown("""
        <div style='font-size: 15px; line-height: 1.8;'>
            <p><strong>Time Series Forecasting:</strong></p>
            <ul style='font-size: 14px;'>
                <li><strong>Pattern Recognition:</strong> 24 months reveals seasonal cycles</li>
                <li><strong>Demand Prediction:</strong> Forecast sales 3-6 months ahead (85-90% accuracy)</li>
                <li><strong>Inventory Planning:</strong> Predict which products needed when</li>
                <li><strong>Anomaly Detection:</strong> Spot unusual patterns early</li>
            </ul>
            <p style='font-size: 14px;'><strong>Models:</strong> ARIMA, Prophet, LSTM Neural Networks</p>
        </div>
        """, unsafe_allow_html=True)

# Department Performance and Customer Segmentation Visualizations

# RIGHT COLUMN: DEPARTMENT PERFORMANCE
with col2:
    st.subheader("Product Department Performance")
    st.markdown("""
    <p style='font-size: 18px; color: #555;'>
        Top 10 departments by revenue help optimize inventory and merchandising.
    </p>
    """, unsafe_allow_html=True)
    
    # Top 10 departments by revenue
    dept_sales = dept_performance.sort_values('TOTAL_REVENUE', ascending=False).head(10)
    dept_sales = dept_sales.sort_values('TOTAL_REVENUE', ascending=True)  # For horizontal bar
    
    fig_dept = go.Figure()
    fig_dept.add_trace(go.Bar(
        y=dept_sales['DEPARTMENT'],
        x=dept_sales['TOTAL_REVENUE'],
        orientation='h',
        marker=dict(
            color=dept_sales['TOTAL_REVENUE'],
            colorscale='Blues',
            showscale=False
        ),
        text=dept_sales['TOTAL_REVENUE'].apply(lambda x: f'${x/1000:.0f}K'),
        textposition='outside',
        textfont=dict(size=14),
        hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<br>Customers: %{customdata}<extra></extra>',
        customdata=dept_sales['NUM_CUSTOMERS']
    ))
    
    fig_dept.update_layout(
        height=400,
        font=dict(size=14),
        xaxis=dict(
            title=dict(
                text='Revenue ($)',
                font=dict(size=16)),
            tickfont=dict(size=12),
            gridcolor='#e0e0e0'
        ),
        yaxis=dict(
            title='',
            tickfont=dict(size=12)
        ),
        plot_bgcolor='white',
        margin=dict(l=100, r=20, t=20, b=60)
    )
    
    st.plotly_chart(fig_dept, use_container_width=True)
    
    # Top department insights
    top_dept = dept_performance.sort_values('TOTAL_REVENUE', ascending=False).iloc[0]
    st.markdown(f"""
    <div style='font-size: 15px; line-height: 1.6;'>
        <h4 style='color: #1f4788; margin-top: 0; font-size: 18px;'>
            Top Performing Department
        </h4>
        <p style='font-size: 16px; margin: 0;'>
            <strong>{top_dept['DEPARTMENT']}</strong>: <strong>${top_dept['TOTAL_REVENUE']:,.0f}</strong>
        </p>
        <p style='font-size: 14px; margin-bottom: 0;'>
            {top_dept['NUM_CUSTOMERS']:,} customers, {top_dept['NUM_BASKETS']:,} trips
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ML explanation
    with st.expander("Click here to see how this data helps Machine Learning", expanded=False):
        st.markdown("""
        <div style='font-size: 15px; line-height: 1.8;'>
            <p><strong>Product Recommendations & Inventory:</strong></p>
            <ul style='font-size: 14px;'>
                <li><strong>Market Basket Analysis:</strong> Which departments purchased together</li>
                <li><strong>Cross-Selling:</strong> Recommend complementary products</li>
                <li><strong>Stock Optimization:</strong> Predict optimal inventory levels</li>
                <li><strong>Shelf Space:</strong> Data-driven store layout decisions</li>
            </ul>
            <p style='font-size: 14px;'><strong>Models:</strong> Association Rules (Apriori, FP-Growth), Collaborative Filtering</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# CUSTOMER SEGMENTATION
st.header("Customer Demographics & Behavior")
st.markdown("""
<p style='font-size: 18px; color: #555;'>
    The 2,500 households show diverse shopping patterns. 
    Understanding these differences enables personalized marketing and targeted offers.
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Demographic Group distribution (classification_1: Group1-Group6)
    demo_group = df.groupby('DEMOGRAPHIC_GROUP')['SALES_VALUE'].sum().reset_index()
    demo_group = demo_group.sort_values('SALES_VALUE', ascending=False)
    demo_group = demo_group[demo_group['DEMOGRAPHIC_GROUP'].notna()]  # Remove NaN
    
    if len(demo_group) > 0:
        fig_demo = px.pie(
            demo_group,
            values='SALES_VALUE',
            names='DEMOGRAPHIC_GROUP',
            title='Revenue by Demographic Group',
            color_discrete_sequence=['#1f4788', '#2c5aa0', '#3d6bb3', '#5a8cc9', '#7ba3d1', '#a0c4e8'],
            hole=0.4
        )
        
        fig_demo.update_traces(
            textposition='outside',
            textfont=dict(size=16),
            hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
        )
        
        fig_demo.update_layout(
            font=dict(size=16),
            title=dict(font=dict(size=20)),
            height=450,
            legend=dict(font=dict(size=14))
        )
        
        st.plotly_chart(fig_demo, use_container_width=True)

with col2:
    # Shopping Segment distribution (classification_5: Group1-Group6)
    shopping_seg = df.groupby('SHOPPING_SEGMENT')['SALES_VALUE'].sum().reset_index()
    shopping_seg = shopping_seg.sort_values('SALES_VALUE', ascending=False)
    shopping_seg = shopping_seg[shopping_seg['SHOPPING_SEGMENT'].notna()]  # Remove NaN
    
    if len(shopping_seg) > 0:
        fig_shop = px.pie(
            shopping_seg,
            values='SALES_VALUE',
            names='SHOPPING_SEGMENT',
            title='Revenue by Shopping Segment',
            color_discrete_sequence=['#2c5aa0', '#3d6bb3', '#5a8cc9', '#7ba3d1', '#a0c4e8', '#c5ddf3'],
            hole=0.4
        )
        
        fig_shop.update_traces(
            textposition='outside',
            textfont=dict(size=16),
            hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
        )
        
        fig_shop.update_layout(
            font=dict(size=16),
            title=dict(font=dict(size=20)),
            height=450,
            legend=dict(font=dict(size=14))
        )
        
        st.plotly_chart(fig_shop, use_container_width=True)

# Customer value distribution
st.subheader("Customer Value Distribution")

# Segment customers by spending
customer_spending = customer_metrics.sort_values('TOTAL_SPENT', ascending=False).reset_index(drop=True)
customer_spending['PERCENTILE'] = (customer_spending.index / len(customer_spending) * 100).astype(int)
customer_spending['VALUE_SEGMENT'] = pd.cut(
    customer_spending['PERCENTILE'],
    bins=[0, 20, 50, 80, 100],
    labels=['Top 20% (VIP)', 'High Value (21-50%)', 'Medium Value (51-80%)', 'Lower Value (81-100%)']
)

segment_summary = customer_spending.groupby('VALUE_SEGMENT').agg({
    'TOTAL_SPENT': 'sum',
    'household_key': 'count'
}).reset_index()
segment_summary.columns = ['VALUE_SEGMENT', 'TOTAL_REVENUE', 'NUM_CUSTOMERS']
segment_summary['PCT_REVENUE'] = (segment_summary['TOTAL_REVENUE'] / segment_summary['TOTAL_REVENUE'].sum() * 100).round(1)

# Display as metrics
col1, col2, col3, col4 = st.columns(4)
for idx, row in segment_summary.iterrows():
    with [col1, col2, col3, col4][idx]:
        st.metric(
            label=str(row['VALUE_SEGMENT']),
            value=f"{row['PCT_REVENUE']:.1f}%"
        )

with st.expander("Click here to learn how this data helps Machine Learning", expanded=False):
    st.markdown("""
    <div style='font-size: 16px; line-height: 1.8;'>
        <p><strong>Customer Segmentation & Personalization:</strong></p>
        <ul>
            <li><strong>RFM Analysis:</strong> Segment by Recency, Frequency, Monetary value</li>
            <li><strong>Churn Prediction:</strong> Identify customers at risk of leaving (85%+ accuracy)</li>
            <li><strong>Lifetime Value:</strong> Predict each customer's future spending potential</li>
            <li><strong>Targeted Campaigns:</strong> Send personalized offers to the right segments</li>
            <li><strong>Look-alike Modeling:</strong> Find new customers similar to your best ones</li>
        </ul>
        <p><strong>Models to use:</strong> K-Means Clustering, Random Forest, XGBoost, Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# SHOPPING TIME PATTERNS
st.header("When Do Customers Shop?")
st.markdown("""
<p style='font-size: 18px; color: #555;'>
    Understanding shopping patterns across 102 weeks 
    helps optimize staffing, promotions, and store operations.
</p>
""", unsafe_allow_html=True)

# Day of week pattern (centered, full width)
dow_pattern = df.groupby('DAY_NAME').agg({
    'SALES_VALUE': 'sum',
    'BASKET_ID': 'nunique',
    'household_key': 'nunique'
}).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

fig_dow = go.Figure()
fig_dow.add_trace(go.Bar(
    x=dow_pattern.index,
    y=dow_pattern['SALES_VALUE'],
    marker=dict(color='#1f4788'),
    text=dow_pattern['SALES_VALUE'].apply(lambda x: f'${x/1000:.0f}K'),
    textposition='outside',
    textfont=dict(size=16),
    hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<br>Shopping Trips: %{customdata[0]:,}<br>Customers: %{customdata[1]:,}<extra></extra>',
    customdata=np.column_stack((dow_pattern['BASKET_ID'], dow_pattern['household_key']))
))

fig_dow.update_layout(
    title=dict(text='Revenue by Day of Week', font=dict(size=24)),
    xaxis=dict(title='', tickfont=dict(size=16)),
    yaxis=dict(
        title=dict(
            text='Revenue ($)',
            font=dict(size=18)  # <-- FIXED: 'titlefont' is now 'font' inside 'title'
        ),
        tickfont=dict(size=14),
        gridcolor='#e0e0e0'
    ),
    height=500,
    font=dict(size=16),
    plot_bgcolor='white'
)

st.plotly_chart(fig_dow, use_container_width=True)

# Weekend vs Weekday comparison
weekend_revenue = df[df['IS_WEEKEND'] == 1]['SALES_VALUE'].sum()
weekday_revenue = df[df['IS_WEEKEND'] == 0]['SALES_VALUE'].sum()
weekend_trips = df[df['IS_WEEKEND'] == 1]['BASKET_ID'].nunique()
weekday_trips = df[df['IS_WEEKEND'] == 0]['BASKET_ID'].nunique()

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Weekday Revenue (Mon-Fri)",
        f"${weekday_revenue:,.0f}",
        f"{weekday_trips:,} shopping trips"
    )
with col2:
    st.metric(
        "Weekend Revenue (Sat-Sun)",
        f"${weekend_revenue:,.0f}",
        f"{weekend_trips:,} shopping trips"
    )

# Peak shopping insights
peak_day = dow_pattern['SALES_VALUE'].idxmax()

st.markdown(f"""
<div style='background-color: #fff9e6; padding: 20px; border-radius: 10px; margin-top: 20px;'>
    <h3 style='color: #1f4788; margin-top: 0; font-size: 22px;'>Peak Shopping Day</h3>
    <p style='font-size: 18px; margin: 10px 0;'>
        <strong>{peak_day}</strong> is the busiest shopping day with <strong>${dow_pattern.loc[peak_day, 'SALES_VALUE']:,.0f}</strong> in sales
    </p>
    <p style='font-size: 16px; margin-bottom: 0;'>
        This day accounts for {dow_pattern.loc[peak_day, 'BASKET_ID']:,} shopping trips from {dow_pattern.loc[peak_day, 'household_key']:,} different customers
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("Clik here to learn how this data helps Machine Learning", expanded=False):
    st.markdown("""
    <div style='font-size: 16px; line-height: 1.8;'>
        <p><strong>Demand Forecasting & Resource Optimization:</strong></p>
        <ul>
            <li><strong>Staff Scheduling:</strong> Predict exactly how many employees needed each hour</li>
            <li><strong>Promotional Timing:</strong> Launch campaigns when customers are most active</li>
            <li><strong>Delivery Planning:</strong> Optimize delivery slots based on demand patterns</li>
            <li><strong>Store Hours:</strong> Data-driven decisions on opening/closing times</li>
        </ul>
        <p><strong>Models to use:</strong> Time Series Regression, Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# CAMPAIGN EFFECTIVENESS (30 campaigns)
st.header("Marketing Campaign Performance")
st.markdown("""
<p style='font-size: 18px; color: #555;'>
    With 30 different campaigns (TypeA, TypeB, TypeC), 
    understanding which work best helps maximize marketing ROI.
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# LEFT COLUMN: Campaign vs Non-Campaign Comparison
with col1:
    st.subheader("Campaign Impact Analysis")
    
    # Compare campaign vs non-campaign customers
    campaign_comparison = df.groupby('IN_CAMPAIGN').agg({
        'SALES_VALUE': 'sum',
        'QUANTITY': 'sum',
        'BASKET_ID': 'nunique',
        'household_key': 'nunique'
    }).reset_index()

    campaign_comparison['IN_CAMPAIGN_LABEL'] = campaign_comparison['IN_CAMPAIGN'].map({
        0: 'No Campaign',
        1: 'In Campaign'
    })

    campaign_comparison['AVG_REVENUE_PER_CUSTOMER'] = (
        campaign_comparison['SALES_VALUE'] / campaign_comparison['household_key']
    )

    # Create comparison visualization
    fig_campaign = go.Figure()

    fig_campaign.add_trace(go.Bar(
        name='Average Revenue per Customer',
        x=campaign_comparison['IN_CAMPAIGN_LABEL'],
        y=campaign_comparison['AVG_REVENUE_PER_CUSTOMER'],
        marker=dict(color=['#7ba3d1', '#1f4788']),
        text=campaign_comparison['AVG_REVENUE_PER_CUSTOMER'].apply(lambda x: f'${x:,.2f}'),
        textposition='outside',
        textfont=dict(size=16, color='black'),
        hovertemplate='<b>%{x}</b><br>Avg Revenue: $%{y:,.2f}<br>Customers: %{customdata:,}<extra></extra>',
        customdata=campaign_comparison['household_key']
    ))

    fig_campaign.update_layout(
        xaxis=dict(title='', tickfont=dict(size=16)),
        yaxis=dict(
            title=dict(
                text='Avg Revenue per Customer ($)',
                font=dict(size=16)
            ),
            tickfont=dict(size=12)
        ),
        height=400,
        font=dict(size=14),
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=60, r=20, t=20, b=60)
    )

    st.plotly_chart(fig_campaign, use_container_width=True)

    # Campaign lift calculation
    if len(campaign_comparison) == 2:
        campaign_revenue = campaign_comparison[
            campaign_comparison['IN_CAMPAIGN'] == 1
        ]['AVG_REVENUE_PER_CUSTOMER'].values[0]
        
        non_campaign_revenue = campaign_comparison[
            campaign_comparison['IN_CAMPAIGN'] == 0
        ]['AVG_REVENUE_PER_CUSTOMER'].values[0]
        
        lift = ((campaign_revenue - non_campaign_revenue) / non_campaign_revenue) * 100
        
        campaign_customers = campaign_comparison[campaign_comparison['IN_CAMPAIGN'] == 1]['household_key'].values[0]
        campaign_total_revenue = campaign_comparison[campaign_comparison['IN_CAMPAIGN'] == 1]['SALES_VALUE'].values[0]
        
        st.markdown(f"""
        <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #1f4788; margin-top: 0; margin-bottom: 10px; font-size: 18px;'>
                Campaign Effectiveness
            </h4>
            <p style='font-size: 16px; margin: 6px 0;'>
                <strong>Lift: {lift:+.1f}%</strong>
            </p>
            <p style='font-size: 14px; margin: 0;'>
                Campaign: <strong>${campaign_revenue:,.2f}</strong> | Non-Campaign: <strong>${non_campaign_revenue:,.2f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

# RIGHT COLUMN: Campaign Type Performance
with col2:
    st.subheader("Campaign Type Breakdown")
    
    campaign_type_perf = df[df['IN_CAMPAIGN'] == 1].groupby('CAMPAIGN_TYPE').agg({
        'SALES_VALUE': 'sum',
        'household_key': 'nunique',
        'BASKET_ID': 'nunique'
    }).reset_index()

    campaign_type_perf['AVG_REVENUE_PER_CUSTOMER'] = (
        campaign_type_perf['SALES_VALUE'] / campaign_type_perf['household_key']
    )

    if len(campaign_type_perf) > 0:
        fig_type = go.Figure()
        fig_type.add_trace(go.Bar(
            x=campaign_type_perf['CAMPAIGN_TYPE'],
            y=campaign_type_perf['AVG_REVENUE_PER_CUSTOMER'],
            marker=dict(color=['#1f4788', '#3d6bb3', '#7ba3d1'][:len(campaign_type_perf)]),
            text=campaign_type_perf['AVG_REVENUE_PER_CUSTOMER'].apply(lambda x: f'${x:,.2f}'),
            textposition='outside',
            textfont=dict(size=16),
            hovertemplate='<b>%{x}</b><br>Avg Revenue: $%{y:,.2f}<br>Customers: %{customdata:,}<extra></extra>',
            customdata=campaign_type_perf['household_key']
        ))
        
        fig_type.update_layout(
            xaxis=dict(
                title=dict(text='Campaign Type', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(text='Avg Revenue per Customer ($)', font=dict(size=16)),
                tickfont=dict(size=12)
            ),
            height=400,
            plot_bgcolor='white',
            margin=dict(l=60, r=20, t=20, b=60)
        )
        
        st.plotly_chart(fig_type, use_container_width=True)
        
        # Best performing campaign type
        best_type = campaign_type_perf.sort_values('AVG_REVENUE_PER_CUSTOMER', ascending=False).iloc[0]
        st.markdown(f"""
        <div style='background-color: #e8f9f0; padding: 20px; border-radius: 10px;'>
            <h4 style='color: #1f4788; margin-top: 0; margin-bottom: 10px; font-size: 18px;'>
                Top Campaign Type
            </h4>
            <p style='font-size: 16px; margin: 6px 0;'>
                <strong>{best_type['CAMPAIGN_TYPE']}</strong>: <strong>${best_type['AVG_REVENUE_PER_CUSTOMER']:,.2f}</strong>
            </p>
            <p style='font-size: 14px; margin: 0;'>
                {best_type['household_key']:,} customers, {best_type['BASKET_ID']:,} trips
            </p>
        </div>
        """, unsafe_allow_html=True)

# ML explanation below both charts (full width)
with st.expander("Click here to see how this data helps Machine Learning", expanded=False):
    st.markdown("""
    <div style='font-size: 16px; line-height: 1.8;'>
        <p><strong>Marketing Optimization & Response Prediction:</strong></p>
        <ul>
            <li><strong>Campaign Response Models:</strong> Predict which customers will respond to each of the 30 campaigns</li>
            <li><strong>Budget Allocation:</strong> Optimize spend across TypeA, TypeB, and TypeC campaigns</li>
            <li><strong>Personalized Offers:</strong> Match the right campaign type to each customer segment</li>
            <li><strong>Uplift Modeling:</strong> Identify customers who will buy MORE because of campaigns</li>
            <li><strong>ROI Prediction:</strong> Forecast return on investment before launching campaigns</li>
        </ul>
        <p><strong>Models to use:</strong> Propensity Scoring, Uplift Trees, Causal ML</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")