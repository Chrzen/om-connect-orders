# utils/plotting.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from scipy import optimize
from config.config import oldmutual_palette, negative_color

# ----------------------------------------
# Plotting Functions with Error Handling
# ----------------------------------------
def safe_plot_daily_kpis(df_tab):
    """Wrapper for daily KPI plot with error handling"""
    try:
        if df_tab.empty or 'retail_amount' not in df_tab.columns:
            st.info("No sales data available to plot daily KPIs.")
            return None
        return plot_daily_kpis(df_tab)
    except Exception as e:
        st.error(f"Error creating daily KPI chart: {e}")
        return None

def safe_plot_product_timeline(df_tab):
    """Wrapper for product type timeline plot with error handling"""
    try:
        if df_tab.empty or 'product_type_desc' not in df_tab.columns:
            st.info("No product data available to plot.")
            return None
        return plot_product_type_timeline(df_tab)
    except Exception as e:
        st.error(f"Error creating product timeline chart: {e}")
        return None

def safe_plot_growth_bars(df_tab):
    """Wrapper for period-over-period growth plot with error handling"""
    try:
        if df_tab.empty or df_tab.shape[0] < 2:
            st.info("Not enough data to calculate growth metrics.")
            return None
        return plot_growth_bars(df_tab)
    except Exception as e:
        st.error(f"Error creating growth bars chart: {e}")
        return None

def safe_plot_period_comparison(df_tab, start_date, end_date):
    """Wrapper for period-over-period comparison plot with error handling"""
    try:
        period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        required_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=period_length + 1)
        if df_tab.empty or pd.to_datetime(df_tab['created_at'].min()) > required_start_date:
            st.info("Not enough historical data for This Period vs. Last Period comparison. Try a shorter date range.")
            return None
        return plot_period_comparison_bars(df_tab, start_date, end_date)
    except Exception as e:
        st.error(f"Error creating period comparison chart: {e}")
        return None

def safe_plot_product_treemap(df_tab):
    """Wrapper for product treemap plot with error handling"""
    try:
        if df_tab.empty or 'product_name' not in df_tab.columns:
            st.info("No product data available to generate a treemap.")
            return None
        return plot_product_treemap(df_tab)
    except Exception as e:
        st.error(f"Error creating product treemap: {e}")
        return None

def safe_plot_weekly_kpis(df_tab):
    """Wrapper for weekly KPI plot with error handling"""
    try:
        if df_tab.empty or 'customer_id_number' not in df_tab.columns:
            st.info("Not enough data to plot weekly KPIs. This chart requires the 'customer_id_number' column.")
            return None
        if df_tab['created_at'].nunique() < 2:
            st.info("Not enough data to calculate weekly growth metrics. Please select a longer date range.")
            return None
        return plot_weekly_kpis(df_tab)
    except Exception as e:
        st.error(f"Error creating weekly KPI dashboard: {e}")
        return None

def safe_plot_bundle_analysis(df_tab):
    """Wrapper for the bundle analysis plot with error handling."""
    try:
        if df_tab.empty or 'product_name' not in df_tab.columns:
            st.info("No product data available to generate bundle analysis.")
            return None
        return plot_bundle_analysis(df_tab)
    except Exception as e:
        st.error(f"Error creating bundle analysis chart: {e}")
        return None

def safe_plot_customer_lifecycle(df_tab):
    """Wrapper for the customer lifecycle plot with error handling."""
    try:
        required_cols = ['sim_msisdn', 'created_at', 'retail_amount', 'product_name']
        if df_tab.empty or any(col not in df_tab.columns for col in required_cols):
            st.info("Lifecycle analysis requires 'sim_msisdn', 'created_at', 'retail_amount', and 'product_name' columns.")
            return None
        return plot_customer_lifecycle(df_tab)
    except Exception as e:
        st.error(f"Error creating customer lifecycle chart: {e}")
        return None

def safe_plot_cohort_analysis(df_tab):
    """Wrapper for the cohort analysis plot with error handling."""
    try:
        required_cols = ['sim_msisdn', 'created_at', 'retail_amount']
        if df_tab.empty or any(col not in df_tab.columns for col in required_cols):
            st.info("Cohort analysis requires 'sim_msisdn', 'created_at', and 'retail_amount' columns.")
            return None
        if df_tab['created_at'].dt.to_period('W').nunique() < 2:
            st.info("Not enough data for cohort analysis. Please select a date range spanning at least two weeks.")
            return None
        return plot_cohort_analysis(df_tab)
    except Exception as e:
        st.error(f"Error creating cohort analysis chart: {e}")
        return None

def safe_plot_geo_demographic_overview(df_tab):
    """Wrapper for the Geo & Demo overview plot with error handling."""
    try:
        required_cols = ['province', 'age_group', 'retail_amount']
        if df_tab.empty or any(col not in df_tab.columns for col in required_cols):
            st.info("Geographic & demographic analysis requires 'province', 'age_group', and 'retail_amount' columns.")
            return None
        return plot_geo_demographic_overview(df_tab)
    except Exception as e:
        st.error(f"Error creating geographic & demographic overview: {e}")
        return None

def safe_plot_correlation_matrix(df_tab):
    """Wrapper for the correlation matrix with error handling."""
    try:
        if df_tab.empty:
            return None
        return plot_correlation_matrix(df_tab)
    except Exception as e:
        st.error(f"Error creating correlation matrix: {e}")
        return None

def safe_plot_product_correlation_matrix(df_tab):
    """Wrapper for the product correlation matrix with error handling."""
    try:
        required_cols = ['sim_msisdn', 'product_name']
        if df_tab.empty or any(col not in df_tab.columns for col in required_cols):
            return None
        return plot_product_correlation_matrix(df_tab)
    except Exception as e:
        st.error(f"Error creating product correlation matrix: {e}")
        return None

def safe_plot_split_sunbursts(df_tab):
    """Wrapper for the split sunburst charts with error handling."""
    try:
        required_cols = ['province', 'age_group', 'product_name', 'retail_amount']
        if df_tab.empty or any(col not in df_tab.columns for col in required_cols):
            return None
        return plot_split_sunbursts(df_tab)
    except Exception as e:
        st.error(f"Error creating split sunburst charts: {e}")
        return None

def safe_plot_purchase_sequence_sankey(df_tab):
    """Wrapper for the purchase sequence Sankey diagram with error handling."""
    try:
        required_cols = ['sim_msisdn', 'created_at', 'product_name']
        if df_tab.empty or any(col not in df_tab.columns for col in required_cols):
            st.info("Purchase sequence analysis requires 'sim_msisdn', 'created_at', and 'product_name' columns.")
            return None
        return plot_purchase_sequence_sankey(df_tab)
    except Exception as e:
        st.error(f"Error creating purchase sequence Sankey diagram: {e}")
        return None

def safe_plot_repurchase_propensity(df_tab):
    """Wrapper for the repurchase propensity curve with error handling."""
    try:
        required_cols = ['sim_msisdn', 'created_at', 'product_name']
        if df_tab.empty or any(col not in df_tab.columns for col in required_cols) or df_tab.groupby('sim_msisdn').size().max() < 2:
            st.info("Repurchase analysis requires multiple purchases per SIM. Not enough data in the selected range.")
            return None
        return plot_repurchase_propensity(df_tab)
    except Exception as e:
        st.error(f"Error creating repurchase propensity curve: {e}")
        return None

# ----------------------------------------
# Plotly Visualization Functions
# ----------------------------------------
def plot_repurchase_propensity(df):
    """
    Analyzes and plots the time between purchases, segmented by the specific
    product that was purchased previously.
    """
    df = df.sort_values(['sim_msisdn', 'created_at']).copy()
    df['time_since_last_purchase'] = df.groupby('sim_msisdn')['created_at'].diff().dt.days
    df['previous_product'] = df.groupby('sim_msisdn')['product_name'].shift(1)
    
    df_repurchases = df.dropna(subset=['time_since_last_purchase', 'previous_product'])

    top_trigger_products = df_repurchases['previous_product'].value_counts().nlargest(7).index
    
    fig = go.Figure()
    colors = px.colors.qualitative.Vivid 

    for i, product in enumerate(top_trigger_products):
        segment_data = df_repurchases[df_repurchases['previous_product'] == product]
        if segment_data.empty:
            continue
        
        ecdf = segment_data['time_since_last_purchase'].value_counts(normalize=True).sort_index().cumsum()
        
        fig.add_trace(go.Scatter(
            x=ecdf.index,
            y=ecdf.values,
            mode='lines',
            name=product,
            line=dict(color=colors[i % len(colors)], width=3),
            hovertemplate=f'<b>{product}</b><br>' + 
                          '%{y:.0%} of next purchases happen within %{x} days<extra></extra>'
        ))

    fig.update_layout(
        title_text="Repurchase Propensity by Preceding Product",
        xaxis_title="Days Since Last Purchase",
        yaxis_title="Probability of Next Purchase Occurring",
        yaxis_tickformat='.0%',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(title="Preceding Product")
    )
    fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.2)')
    
    fig.add_annotation(
        text="This shows how quickly customers buy again after purchasing a specific product.",
        align='left',
        showarrow=False,
        xref='paper', yref='paper',
        x=0.02, y=-0.20,
        font_color="grey"
    )

    return fig

def plot_purchase_sequence_sankey(df):
    """
    Creates a Sankey diagram to visualize the sequence of product purchases
    for the first 4 purchases.
    """
    def hex_to_rgba(hex_color, alpha=0.6):
        """Convert hex color to rgba with transparency"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    
    df = df.sort_values(['sim_msisdn', 'created_at']).copy()

    df['purchase_rank'] = df.groupby('sim_msisdn').cumcount() + 1

    max_rank = 4
    df_sequence = df[df['purchase_rank'] <= max_rank]
    top_products = df_sequence['product_name'].value_counts().nlargest(8).index
    df_sequence['product_name_agg'] = df_sequence['product_name'].apply(lambda x: x if x in top_products else 'Other')

    df_sequence['next_product'] = df_sequence.groupby('sim_msisdn')['product_name_agg'].shift(-1)
    df_sequence.dropna(subset=['next_product'], inplace=True)

    ordinal_map = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}
    df_sequence['source_label'] = df_sequence['product_name_agg'] + ' (' + df_sequence['purchase_rank'].map(ordinal_map) + ')'
    df_sequence['target_label'] = df_sequence['next_product'] + ' (' + (df_sequence['purchase_rank'] + 1).map(ordinal_map) + ')'

    links = df_sequence.groupby(['source_label', 'target_label']).size().reset_index(name='value')
    links = links[links['value'] > links['value'].quantile(0.5)]

    all_nodes = pd.unique(links[['source_label', 'target_label']].values.ravel('K'))
    node_map = {node: i for i, node in enumerate(all_nodes)}

    link_colors = []
    for src in links['source_label']:
        if '(1st)' in src:
            link_colors.append(hex_to_rgba(oldmutual_palette[1], 0.6))
        elif '(2nd)' in src:
            link_colors.append(hex_to_rgba(oldmutual_palette[2], 0.6))
        elif '(3rd)' in src:
            link_colors.append(hex_to_rgba(oldmutual_palette[3], 0.6))
        else:
            link_colors.append(hex_to_rgba(oldmutual_palette[4], 0.6))

    link_data = dict(
        source=[node_map[src] for src in links['source_label']],
        target=[node_map[tgt] for tgt in links['target_label']],
        value=links['value'],
        color=link_colors
    )

    # Change: Use darker node colors
    node_colors = []
    dark_palette = [oldmutual_palette[0], oldmutual_palette[2]]
    for i, node in enumerate(all_nodes):
        node_colors.append(dark_palette[i % len(dark_palette)])

    node_data = dict(
        label=all_nodes,
        pad=15,
        thickness=20,
        color=node_colors,  # Using the new darker palette
        line=dict(color="white", width=1)
    )

    fig = go.Figure(go.Sankey(
        node=node_data,
        link=link_data,
        arrangement='snap'
    ))

    fig.update_layout(
        title_text="Customer Purchase Journey: Which Product Comes Next?",
        font_size=12,
        height=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    return fig

def plot_split_sunbursts(df):
    """
    Creates two side-by-side interactive sunburst charts for revenue analysis:
    1. By Province -> Product Category
    2. By Age Group -> Product Category
    """
    df_filtered = df[['province', 'age_group', 'product_name', 'retail_amount']].dropna()
    df_filtered['product_category'] = df_filtered['product_name'].apply(
        lambda x: 'Airtime' if 'airtime' in x.lower() else ('Data' if any(s in x for s in ['MB', 'GB']) else 'Other')
    )

    # CORRECTED: Explicitly map categories to the desired palette colors
    color_map = {
        'Data': oldmutual_palette[1],
        'Airtime': oldmutual_palette[2],
        'Other': oldmutual_palette[4]
    }

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=("Revenue by Province", "Revenue by Age Group")
    )

    # Subplot 1: By Province
    px_fig_province = px.sunburst(
        df_filtered, path=['province', 'product_category'], values='retail_amount',
        color="product_category",
        color_discrete_map=color_map # Use the explicit map
    )
    fig.add_trace(px_fig_province.data[0], row=1, col=1)

    # Subplot 2: By Age
    px_fig_age = px.sunburst(
        df_filtered, path=['age_group', 'product_category'], values='retail_amount',
        color="product_category",
        color_discrete_map=color_map # Use the explicit map
    )
    fig.add_trace(px_fig_age.data[0], row=1, col=2)

    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(t=60, l=10, r=10, b=10)
    )
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Total Revenue: R%{value:,.2f}<extra></extra>'
    )

    return fig

def plot_daily_kpis(df):
    """
    Creates two separate plots stacked vertically:
    1. Daily revenue (line chart with area fill)
    2. 7-day rolling revenue growth metric (bars)
    """
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    daily_kpis = df.resample('D', on='created_at').agg(
        total_revenue=('retail_amount', 'sum')
    ).reset_index()

    # Calculate a 7-day rolling average and its percentage change
    daily_kpis['revenue_7d_avg'] = daily_kpis['total_revenue'].rolling(window=7).mean()
    daily_kpis['revenue_growth'] = daily_kpis['revenue_7d_avg'].pct_change() * 100

    # CHANGE: Split into two separate plots
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.15
    )

    # Plot 1: Daily Total Revenue (Line chart with area fill)
    fig.add_trace(go.Scatter(
        x=daily_kpis['created_at'], y=daily_kpis['total_revenue'], name="Daily Revenue",
        line=dict(color='#5AAA46'), fill='tozeroy', fillcolor='rgba(90, 170, 70, 0.2)',
        hovertemplate='<b>%{x}</b><br>Revenue: R%{y:,.2f}<extra></extra>'
    ), row=1, col=1)

    # Plot 2: Growth as bars
    fig.add_trace(go.Bar(
        x=daily_kpis['created_at'], y=daily_kpis['revenue_growth'], name="7-Day Growth Trend",
        marker_color=np.where(daily_kpis['revenue_growth'] < 0, negative_color, "#8CC63F"),
        hovertemplate='Growth: %{y:.2f}%<extra></extra>'
    ), row=2, col=1)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="", gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="<b>Daily Revenue (R)</b>", row=1, col=1, tickformat="$,.0f", gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="<b>7-Day Avg. Growth (%)</b>", row=2, col=1, gridcolor='rgba(255, 255, 255, 0.2)', ticksuffix="%")
    
    return fig

def plot_weekly_kpis(df):
    """Plot multiple KPIs on a weekly basis (orders, revenue, customers) using Plotly."""
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])

    weekly_agg = df.resample('W-SUN', on='created_at').agg(
        order_count=('id', 'size'),
        total_revenue=('retail_amount', 'sum'),
        unique_customers=('customer_id_number', 'nunique')
    ).reset_index()

    if len(weekly_agg) > 1:
        weekly_agg = weekly_agg.iloc[:-1]

    if weekly_agg.empty:
        return None

    weekly_agg['order_growth'] = weekly_agg['order_count'].pct_change()
    weekly_agg['revenue_growth'] = weekly_agg['total_revenue'].pct_change()
    weekly_agg['customer_growth'] = weekly_agg['unique_customers'].pct_change()

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Weekly Order Volume', 'Weekly Revenue', 'Weekly Unique Customers',
            'Weekly Order Growth', 'Weekly Revenue Growth', 'Weekly Customer Growth'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    # CHANGE: Use same green color (#5AAA46) for all timeline areas
    timeline_color = '#5AAA46'

    fig.add_trace(go.Scatter(
        x=weekly_agg['created_at'], y=weekly_agg['order_count'], name="Orders",
        line=dict(color=timeline_color), fill='tozeroy', mode='lines+markers',
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Orders: %{y:,}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=weekly_agg['created_at'], y=weekly_agg['total_revenue'], name="Revenue",
        line=dict(color=timeline_color), fill='tozeroy', mode='lines+markers',
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Revenue: R%{y:,.2f}<extra></extra>'
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=weekly_agg['created_at'], y=weekly_agg['unique_customers'], name="Customers",
        line=dict(color=timeline_color), fill='tozeroy', mode='lines+markers',
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Customers: %{y:,}<extra></extra>'
    ), row=1, col=3)

    fig.add_trace(go.Bar(
        x=weekly_agg['created_at'], y=weekly_agg['order_growth'], name="Order Growth",
        marker_color=[timeline_color if v >= 0 else negative_color for v in weekly_agg['order_growth'].fillna(0)],
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Growth: %{y:.1%}<extra></extra>',
        opacity=0.6
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=weekly_agg['created_at'], y=weekly_agg['revenue_growth'], name="Revenue Growth",
        marker_color=[timeline_color if v >= 0 else negative_color for v in weekly_agg['revenue_growth'].fillna(0)],
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Growth: %{y:.1%}<extra></extra>',
        opacity=0.6
    ), row=2, col=2)

    fig.add_trace(go.Bar(
        x=weekly_agg['created_at'], y=weekly_agg['customer_growth'], name="Customer Growth",
        marker_color=[timeline_color if v >= 0 else negative_color for v in weekly_agg['customer_growth'].fillna(0)],
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Growth: %{y:.1%}<extra></extra>',
        opacity=0.6
    ), row=2, col=3)

    fig.update_layout(
        height=700,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_x=0.5,
        margin=dict(t=80, b=50, l=50, r=50)
    )

    fig.update_yaxes(row=1, col=1, title_text='Number of Orders', gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(row=1, col=2, title_text='Revenue (R)', tickformat="$,.0f", gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(row=1, col=3, title_text='Unique Customers', gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(row=2, col=1, title_text='% Change', tickformat='.1%', zeroline=True, zerolinewidth=1, zerolinecolor='grey', gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(row=2, col=2, title_text='% Change', tickformat='.1%', zeroline=True, zerolinewidth=1, zerolinecolor='grey', gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(row=2, col=3, title_text='% Change', tickformat='.1%', zeroline=True, zerolinewidth=1, zerolinecolor='grey', gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_xaxes(showgrid=False, tickformat="%b %d", tickangle=45)

    return fig


def plot_product_type_timeline(df):
    """
    Creates two separate plots stacked vertically:
    1. Stacked area chart of daily order volume by product type
    2. Bar chart of the 7-day rolling average order growth
    """
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Data preparation
    daily_products = df.groupby([df['created_at'].dt.date, 'product_type_desc']).size().reset_index(name='count')
    daily_products.rename(columns={'created_at': 'date'}, inplace=True)

    daily_totals = daily_products.groupby('date')['count'].sum().reset_index()
    daily_totals['orders_7d_avg'] = daily_totals['count'].rolling(window=7).mean()
    daily_totals['orders_growth'] = daily_totals['orders_7d_avg'].pct_change() * 100

    # CHANGE: Split into two separate plots
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.15
    )
    
    unique_product_types = daily_products['product_type_desc'].unique()

    # Plot 1: Stacked area chart for product types (using oldmutual_palette[0] and oldmutual_palette[2])
    colors_to_use = [oldmutual_palette[0], oldmutual_palette[2]]
    
    for i, ptype in enumerate(unique_product_types):
        plot_df = daily_products[daily_products['product_type_desc'] == ptype]
        color = colors_to_use[i % len(colors_to_use)]
        
        fig.add_trace(go.Scatter(
            x=plot_df['date'],
            y=plot_df['count'],
            name=ptype,
            mode='lines',
            line=dict(width=0.5, color=color),
            stackgroup='one',
            hovertemplate=f'<b>{ptype}</b><br>Date: %{{x}}<br>Orders: %{{y}}<extra></extra>'
        ), row=1, col=1)

    # Plot 2: Bar chart for growth metric
    fig.add_trace(go.Bar(
        x=daily_totals['date'],
        y=daily_totals['orders_growth'],
        name='7-Day Order Growth',
        marker_color=np.where(daily_totals['orders_growth'] < 0, negative_color, "#8CC63F"),
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Growth: %{y:.2f}%<extra></extra>'
    ), row=2, col=1)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.2)')
    
    # Configure axes for both plots
    fig.update_yaxes(
        title_text="<b>Daily Order Count</b>", 
        title_font=dict(color='white'),
        row=1, col=1,
        gridcolor='rgba(255, 255, 255, 0.2)'
    )
    fig.update_yaxes(
        title_text="<b>7-Day Avg. Growth (%)</b>", 
        title_font=dict(color='white'),
        row=2, col=1,
        gridcolor='rgba(255, 255, 255, 0.2)',
        ticksuffix="%"
    )
    
    return fig

def plot_growth_bars(df):
    """
    Calculates and plots WoW and MoM growth for revenue and orders.
    """
    df['created_at'] = pd.to_datetime(df['created_at'])
    end_date = df['created_at'].max()

    current_week_start = end_date - pd.Timedelta(days=6)
    prev_week_end = current_week_start - pd.Timedelta(days=1)
    prev_week_start = prev_week_end - pd.Timedelta(days=6)

    current_month_start = end_date - pd.Timedelta(days=29)
    prev_month_end = current_month_start - pd.Timedelta(days=1)
    prev_month_start = prev_month_end - pd.Timedelta(days=29)

    current_week_df = df[(df['created_at'] >= current_week_start) & (df['created_at'] <= end_date)]
    prev_week_df = df[(df['created_at'] >= prev_week_start) & (df['created_at'] <= prev_week_end)]
    current_month_df = df[(df['created_at'] >= current_month_start) & (df['created_at'] <= end_date)]
    prev_month_df = df[(df['created_at'] >= prev_month_start) & (df['created_at'] <= prev_month_end)]

    metrics = {
        'Revenue (WoW)': {'current': current_week_df['retail_amount'].sum(), 'previous': prev_week_df['retail_amount'].sum()},
        'Orders (WoW)': {'current': current_week_df.shape[0], 'previous': prev_week_df.shape[0]},
        'Revenue (MoM)': {'current': current_month_df['retail_amount'].sum(), 'previous': prev_month_df['retail_amount'].sum()},
        'Orders (MoM)': {'current': current_month_df.shape[0], 'previous': prev_month_df.shape[0]}
    }

    growth_data = {}
    for key, values in metrics.items():
        if values['previous'] > 0:
            growth = (values['current'] - values['previous']) / values['previous']
        elif values['current'] > 0:
            growth = 1.0
        else:
            growth = 0.0
        growth_data[key] = growth

    labels = list(growth_data.keys())
    values = list(growth_data.values())
    colors = ['#8CC63F' if v >= 0 else negative_color for v in values]

    fig = go.Figure(go.Bar(
        x=labels, y=values, text=[f'{v:.1%}' for v in values],
        textposition='outside', marker_color=colors, textfont=dict(size=14, color='white'),
        opacity=0.6
    ))

    fig.update_layout(
        title='Recent Growth (WoW & MoM)',
        yaxis_title='Percentage Change', yaxis_tickformat='.0%',
        xaxis_title=None,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500
    )
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey', gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_xaxes(showgrid=False)
    return fig

def plot_period_comparison_bars(df, current_start, current_end):
    """
    Calculates and plots This Period vs. Last Period growth for the selected date range.
    """
    current_start = pd.to_datetime(current_start).date()
    current_end = pd.to_datetime(current_end).date()
    df['created_at_date'] = pd.to_datetime(df['created_at']).dt.date

    period_length = current_end - current_start
    prev_end = current_start - pd.Timedelta(days=1)
    prev_start = prev_end - period_length

    current_df = df[(df['created_at_date'] >= current_start) & (df['created_at_date'] <= current_end)]
    prev_df = df[(df['created_at_date'] >= prev_start) & (df['created_at_date'] <= prev_end)]

    metrics = {
        f'Revenue ({period_length.days+1}d)': {'current': current_df['retail_amount'].sum(), 'previous': prev_df['retail_amount'].sum()},
        f'Orders ({period_length.days+1}d)': {'current': current_df.shape[0], 'previous': prev_df.shape[0]},
    }

    growth_data = {}
    for key, values in metrics.items():
        if values['previous'] > 0:
            growth = (values['current'] - values['previous']) / values['previous']
        elif values['current'] > 0:
            growth = 1.0
        else:
            growth = 0.0
        growth_data[key] = growth

    labels = list(growth_data.keys())
    values = list(growth_data.values())
    colors = ['#8CC63F' if v >= 0 else negative_color for v in values]

    fig = go.Figure(go.Bar(
        x=labels, y=values, text=[f'{v:.1%}' for v in values],
        textposition='outside', marker_color=colors, textfont=dict(size=14, color='white'),
        opacity=0.6
    ))

    fig.update_layout(
        title='This Period vs. Last Period Growth',
        yaxis_title='Percentage Change',
        yaxis_tickformat='.0%',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=450
    )
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey', gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_xaxes(showgrid=False)
    return fig


def plot_bundle_analysis(df):
    """
    Creates a two-plot analysis of specific data, voice, and SMS bundles,
    showing distribution vs. revenue and weekly popularity.
    """
    product_order = [
        '30 MB', '50 MB', '100 MB', '500 MB', '1 GB', '2 GB', '3 GB', '5 GB', '10 GB (30 Days)', '20 GB (30 Days)',
        'WhatsApp Daily', 'WhatsApp Weekly', 'WhatsApp Monthly',
        'Voice 30 Min', 'Voice 100 Min',
        '30 SMS', '100 SMS', '500 SMS'
    ]

    df_bundles = df[df['product_name'].isin(product_order)].copy()

    if df_bundles.empty:
        st.info("No specific bundle products (e.g., '1 GB', '100 MB') found in the selected data.")
        return None

    df_bundles['product_name'] = pd.Categorical(df_bundles['product_name'], categories=product_order, ordered=True)
    df_bundles.sort_values('product_name', inplace=True)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
        subplot_titles=("Bundle Orders vs. Revenue", "Weekly Bundle Popularity")
    )

    bundle_summary = df_bundles.groupby('product_name', observed=True).agg(
        order_count=('id', 'count'),
        total_revenue=('retail_amount', 'sum')
    ).reset_index()

    # Change 1: Bar color for Bundle Orders
    fig.add_trace(go.Bar(
        x=bundle_summary['product_name'],
        y=bundle_summary['order_count'],
        name='Orders',
        marker_color=oldmutual_palette[1], # Changed color
        hovertemplate='<b>%{x}</b><br>Orders: %{y:,}<extra></extra>'
    ), secondary_y=False, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bundle_summary['product_name'],
        y=bundle_summary['total_revenue'],
        name='Revenue',
        line=dict(color='#8CC63F', width=3),
        hovertemplate='Revenue: R%{y:,.2f}<extra></extra>'
    ), secondary_y=True, row=1, col=1)

    df_bundles['created_at'] = pd.to_datetime(df_bundles['created_at'])

    weekly_counts = df_bundles.groupby([
        pd.Grouper(key='created_at', freq='W-SUN'),
        'product_name'
    ], observed=True).size().unstack(fill_value=0)

    # Change 2: Darker color palette for weekly popularity
    colors = oldmutual_palette
    for i, product in enumerate(weekly_counts.columns):
        fig.add_trace(go.Bar(
            x=weekly_counts.index,
            y=weekly_counts[product],
            name=product,
            marker_color=colors[i % len(colors)],
            hovertemplate=f'<b>{product}</b><br>Week of %{{x|%d %b}}<br>Orders: %{{y}}<extra></extra>'
        ), row=1, col=2)

    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend_traceorder="normal",
        barmode='stack',
        legend=dict(font=dict(size=10))
    )

    fig.update_yaxes(title_text="<b>Order Count</b>", secondary_y=False, row=1, col=1, gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="<b>Total Revenue (R)</b>", secondary_y=True, row=1, col=1, showgrid=False)
    fig.update_xaxes(tickangle=45, row=1, col=1)

    fig.update_yaxes(title_text="<b>Weekly Orders</b>", row=1, col=2, gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_xaxes(title_text="<b>Week</b>", tickformat="%d %b %Y", row=1, col=2)

    return fig


def plot_customer_lifecycle(df):
    """
    Creates visualizations showing customer behavior and value across their lifecycle,
    based on unique SIMs.
    """
    df = df.copy()

    df['created_at'] = pd.to_datetime(df['created_at'])
    df['acquisition_date'] = df.groupby('sim_msisdn')['created_at'].transform('min')
    df['days_since_acquisition'] = (df['created_at'] - df['acquisition_date']).dt.days

    df['lifecycle_stage'] = pd.cut(
        df['days_since_acquisition'],
        bins=[-1, 7, 30, 90, 180, 365, float('inf')],
        labels=['First Week', '8-30 Days', '31-90 Days', '91-180 Days', '181-365 Days', '365+ Days'],
        right=True
    )

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"secondary_y": True}, {}],
               [{"colspan": 2}, None]],
        subplot_titles=(
            "Value & Frequency by Lifecycle Stage",
            "Top Product Preference by Stage",
            "Cumulative Revenue by SIM Tenure"
        ),
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )

    stage_metrics = df.groupby('lifecycle_stage', observed=True).agg(
        avg_order_value=('retail_amount', 'mean'),
        num_orders=('id', 'count'),
        num_sims=('sim_msisdn', pd.Series.nunique)
    )
    stage_metrics['orders_per_sim'] = stage_metrics['num_orders'] / stage_metrics['num_sims']

    fig.add_trace(go.Bar(
        x=stage_metrics.index, y=stage_metrics['avg_order_value'], name='Avg Order Value',
        marker_color=oldmutual_palette[0],
        hovertemplate='<b>%{x}</b><br>Avg Order Value: R%{y:,.2f}<extra></extra>',
        opacity=0.6
    ), secondary_y=False, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=stage_metrics.index, y=stage_metrics['orders_per_sim'], name='Orders per SIM',
        line=dict(color="#8CC63F", width=3),
        hovertemplate='Orders per SIM: %{y:.2f}<extra></extra>'
    ), secondary_y=True, row=1, col=1)

    top_products = df['product_name'].value_counts().nlargest(8).index
    df_top_products = df[df['product_name'].isin(top_products)]

    product_stage_cross = pd.crosstab(
        df_top_products['lifecycle_stage'],
        df_top_products['product_name'],
        normalize='index'
    )

    fig.add_trace(go.Heatmap(
        z=product_stage_cross.values,
        x=product_stage_cross.columns,
        y=product_stage_cross.index,
        colorscale=[oldmutual_palette[0], oldmutual_palette[3]],
        hovertemplate='<b>Stage:</b> %{y}<br><b>Product:</b> %{x}<br><b>Preference:</b> %{z:.1%}<extra></extra>'
    ), row=1, col=2)

    df['week_since_acquisition'] = (df['days_since_acquisition'] // 7)
    weekly_revenue = df.groupby('week_since_acquisition')['retail_amount'].sum()
    cumulative_revenue = weekly_revenue.cumsum()

    fig.add_trace(go.Scatter(
        x=cumulative_revenue.index, y=cumulative_revenue.values, name='Cumulative Revenue',
        line=dict(color=oldmutual_palette[2]), fill='tozeroy',
        hovertemplate='<b>Week %{x}</b><br>Cumulative Revenue: R%{y:,.2f}<extra></extra>'
    ), row=2, col=1)

    for week in [4, 8, 12, 16, 20, 24, 52]:
        if week in cumulative_revenue.index:
            fig.add_vline(x=week, line_width=1, line_dash="dash", line_color="grey", row=2, col=1)
            fig.add_annotation(x=week, y=cumulative_revenue.max()*0.9, text=f"M{week//4}", showarrow=False, bgcolor="#8CC63F", font_color="black", row=2, col=1)

    fig.update_layout(
        height=800,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False
    )
    fig.update_yaxes(title_text="<b>Avg Order Value (R)</b>", row=1, col=1, secondary_y=False, gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="<b>Orders per SIM</b>", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_xaxes(title_text="Weeks Since Acquisition", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Revenue (R)", tickformat="R,.0f", row=2, col=1, gridcolor='rgba(255, 255, 255, 0.2)')

    return fig

def plot_cohort_analysis(df):
    """
    Creates visualizations for weekly cohort analysis, focusing on cumulative revenue
    and average customer value growth. Aggregated by SIM.
    """
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])

    if not df.empty:
        last_full_week_end = df['created_at'].max().normalize() - pd.to_timedelta(df['created_at'].max().dayofweek + 1, unit='d')
        df = df[df['created_at'] <= last_full_week_end].copy()

    if df.empty:
        return None

    df['acquisition_date'] = df.groupby('sim_msisdn')['created_at'].transform('min')
    df['cohort_week'] = df['acquisition_date'].dt.to_period('W').apply(lambda p: p.start_time)
    df['activity_week'] = df['created_at'].dt.to_period('W').apply(lambda p: p.start_time)
    df['cohort_week_num'] = (df['activity_week'] - df['cohort_week']).dt.days // 7

    fig = make_subplots(
        rows=1, cols=2
    )

    cohort_revenue = df.groupby(['cohort_week', 'cohort_week_num'])['retail_amount'].sum().unstack(fill_value=0)
    revenue_cumulative = cohort_revenue.cumsum(axis=1)

    if not revenue_cumulative.empty:
        top_5_cohorts = revenue_cumulative.iloc[:, -1].nlargest(5).index
        # CHANGE: Use darker colors for cohort lines
        colors = [oldmutual_palette[0], oldmutual_palette[1], oldmutual_palette[2], oldmutual_palette[4], '#00A09A'] # Using a slightly different color for the 5th to ensure visibility

        for i, cohort in enumerate(top_5_cohorts):
            cohort_data = revenue_cumulative.loc[cohort]
            fig.add_trace(go.Scatter(
                x=cohort_data.index, y=cohort_data.values,
                name=f"Cohort: {pd.to_datetime(cohort).strftime('%d %b %Y')}",
                line=dict(color=colors[i % len(colors)]),
                mode='lines+markers',
                hovertemplate='<b>Week %{x}</b><br>Revenue: R%{y:,.2f}<extra></extra>'
            ), row=1, col=1)

    weekly_total_revenue = df.groupby('cohort_week_num')['retail_amount'].sum()
    total_sims = df['sim_msisdn'].nunique()

    if total_sims == 0:
        return fig

    avg_cumulative_revenue = weekly_total_revenue.cumsum() / total_sims

    fig.add_trace(go.Scatter(
        x=avg_cumulative_revenue.index, y=avg_cumulative_revenue.values, name='Observed Value',
        line=dict(color=oldmutual_palette[1]), fill='tozeroy',
        hovertemplate='<b>Week %{x}</b><br>Avg Value: R%{y:,.2f}<extra></extra>'
    ), row=1, col=2)

    if len(avg_cumulative_revenue) > 4:
        def log_func(x, a, b, c):
            return a * np.log(b * x + 1) + c
        
        x_data = avg_cumulative_revenue.index.values[1:]
        y_data = avg_cumulative_revenue.values[1:]

        try:
            popt, _ = optimize.curve_fit(log_func, x_data, y_data, maxfev=5000, p0=[1, 0.1, 1])
            
            projection_weeks = 52
            x_projection = np.arange(1, projection_weeks + 1)
            y_projection = log_func(x_projection, *popt)

            fig.add_trace(go.Scatter(
                x=x_projection, y=y_projection, name='Projected Value',
                line=dict(color='#FFFFFF', dash='dash'),
                hovertemplate='Projected Week %{x}<br>Avg Value: R%{y:,.2f}<extra></extra>'
            ), row=1, col=2)

            year_value = y_projection[-1]
            fig.add_annotation(
                x=projection_weeks, y=year_value, text=f"Projected 1-Year LTV<br><b>R{year_value:,.2f}</b>",
                showarrow=True, arrowhead=2, arrowcolor="white", ax=-40, ay=-40,
                bgcolor="#006B54", bordercolor="white", borderwidth=1, row=1, col=2
            )
        except RuntimeError:
            st.warning("Could not generate LTV projection for the selected date range.")

    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100)
    )
    fig.update_xaxes(title_text="Weeks Since Acquisition", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Revenue (R)", tickformat="$,.0f", row=1, col=1, gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_xaxes(title_text="Weeks Since Acquisition", row=1, col=2)
    fig.update_yaxes(title_text="Avg Cumulative Value per SIM (R)", tickformat="$,.0f", row=1, col=2, gridcolor='rgba(255, 255, 255, 0.2)')

    return fig

def plot_geo_demographic_overview(df):
    """
    Creates a side-by-side, dual-axis view of metrics by province and by age group.
    - Left Axis (Bars): Total Revenue
    - Right Axis (Line): Orders per unique SIM
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=("Metrics by Province", "Metrics by Age Group"),
        # --- CHANGE HERE: Increased horizontal spacing to prevent y-axis titles from clashing ---
        horizontal_spacing=0.1 
    )

    # --- Plot 1: Metrics by Province ---
    province_metrics = df.groupby('province').agg(
        total_revenue=('retail_amount', 'sum'),
        order_count=('id', 'count'),
        sim_count=('sim_msisdn', 'nunique')
    ).nlargest(10, 'total_revenue')
    province_metrics['orders_per_sim'] = province_metrics['order_count'] / province_metrics['sim_count']

    # (Reverted) Bar for Total Revenue
    fig.add_trace(go.Bar(
        x=province_metrics.index, y=province_metrics['total_revenue'],
        name='Total Revenue', marker_color=oldmutual_palette[2],
        hovertemplate='<b>%{x}</b><br>Revenue: R%{y:,.2f}<extra></extra>',
        opacity=0.6
    ), secondary_y=False, row=1, col=1)

    # (Reverted) Line for Orders per SIM
    fig.add_trace(go.Scatter(
        x=province_metrics.index, y=province_metrics['orders_per_sim'],
        name='Orders per SIM', line=dict(color="#8CC63F", width=3),
        hovertemplate='Orders per SIM: %{y:.2f}<extra></extra>'
    ), secondary_y=True, row=1, col=1)

    # --- Plot 2: Metrics by Age Group ---
    age_metrics = df.groupby('age_group').agg(
        total_revenue=('retail_amount', 'sum'),
        order_count=('id', 'count'),
        sim_count=('sim_msisdn', 'nunique')
    ).sort_values('total_revenue', ascending=False)
    age_metrics['orders_per_sim'] = age_metrics['order_count'] / age_metrics['sim_count']

    # Bar for Total Revenue
    fig.add_trace(go.Bar(
        x=age_metrics.index, y=age_metrics['total_revenue'],
        name='Total Revenue', marker_color=oldmutual_palette[3],
        hovertemplate='<b>%{x}</b><br>Revenue: R%{y:,.2f}<extra></extra>',
        opacity=0.6
    ), secondary_y=False, row=1, col=2)

    # Line for Orders per SIM
    fig.add_trace(go.Scatter(
        x=age_metrics.index, y=age_metrics['orders_per_sim'],
        name='Orders per SIM', line=dict(color="#8CC63F", width=3),
        hovertemplate='Orders per SIM: %{y:.2f}<extra></extra>'
    ), secondary_y=True, row=1, col=2)

    # --- Layout (Reverted to original) ---
    fig.update_layout(
        height=500, showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    fig.update_yaxes(title_text="Total Revenue (R)", secondary_y=False, row=1, col=1, gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="Orders per SIM", secondary_y=True, row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="Total Revenue (R)", secondary_y=False, row=1, col=2, gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="Orders per SIM", secondary_y=True, row=1, col=2, showgrid=False)
    
    return fig

def plot_product_correlation_matrix(df):
    """
    Calculates and displays a correlation matrix showing which products are
    often purchased by the same customers.
    """
    top_12_products = df['product_name'].value_counts().nlargest(12).index
    df_top_products = df[df['product_name'].isin(top_12_products)]

    purchase_matrix = pd.crosstab(df_top_products['sim_msisdn'], df_top_products['product_name'])
    purchase_matrix_binary = (purchase_matrix > 0).astype(int)

    product_correlation = purchase_matrix_binary.corr()

    # CORRECTED: Use a custom diverging colorscale with the oldmutual_palette
    fig = go.Figure(go.Heatmap(
        z=product_correlation.values,
        x=product_correlation.columns,
        y=product_correlation.columns,
        colorscale=[[0.0, negative_color], [0.5, "#e6e6e6"], [1.0, oldmutual_palette[0]]],
        zmid=0, # Center the colorscale on zero
        text=product_correlation.round(2).values,
        texttemplate="%{text}",
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title_text='Product Purchase Correlation (Top 12 Products)',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_showgrid=False, yaxis_showgrid=False
    )
    fig.update_xaxes(tickangle=30)
    return fig

def plot_hierarchical_sunburst(df):
    """
    Creates an interactive sunburst chart to show hierarchical spending patterns.
    """
    top_provinces = df['province'].value_counts().nlargest(7).index
    df_filtered = df[df['province'].isin(top_provinces)]

    df_filtered = df_filtered[['province', 'age_group', 'product_name', 'retail_amount']].dropna()

    df_filtered['product_category'] = df_filtered['product_name'].apply(
        lambda x: 'Airtime' if 'airtime' in x.lower() else ('Data' if any(s in x for s in ['MB', 'GB']) else 'Other')
    )

    fig = px.sunburst(
        df_filtered,
        path=['province', 'age_group', 'product_category'],
        values='retail_amount',
        color='retail_amount',
        color_continuous_scale=px.colors.sequential.Aggrnyl,
        title="Hierarchical View of Revenue: Province -> Age Group -> Product"
    )

    fig.update_layout(
        height=600,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Total Revenue: R%{value:,.2f}<extra></extra>'
    )
    return fig

def plot_product_correlation_matrix(df):
    """
    Calculates and displays a correlation matrix showing which products are
    often purchased by the same customers.
    """
    top_12_products = df['product_name'].value_counts().nlargest(12).index
    df_top_products = df[df['product_name'].isin(top_12_products)]

    purchase_matrix = pd.crosstab(df_top_products['sim_msisdn'], df_top_products['product_name'])
    purchase_matrix_binary = (purchase_matrix > 0).astype(int)

    product_correlation = purchase_matrix_binary.corr()

    # CORRECTED: Use a custom diverging colorscale with the oldmutual_palette
    fig = go.Figure(go.Heatmap(
        z=product_correlation.values,
        x=product_correlation.columns,
        y=product_correlation.columns,
        colorscale=[[0.0, negative_color], [0.5, "#e6e6e6"], [1.0, oldmutual_palette[0]]],
        zmid=0, # Center the colorscale on zero
        text=product_correlation.round(2).values,
        texttemplate="%{text}",
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title_text='Product Purchase Correlation (Top 12 Products)',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_showgrid=False, yaxis_showgrid=False
    )
    fig.update_xaxes(tickangle=30)
    return fig