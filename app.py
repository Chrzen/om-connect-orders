import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # Imported for the area chart
import numpy as np
import json
import datetime
from scipy import optimize

TABLE_NAME = "APP_SCHEMA.ORDER_DATA_10062025"

# Import Snowflake connector with proper error handling
try:
    import snowflake.connector as sf
except ImportError as e:
    st.error("Snowflake connector not properly installed. Please run: pip install snowflake-connector-python[pandas]")
    st.stop()

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Order Data Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# styling
# styling
# styling
st.markdown("""
<style>
    /* --- GEMINI FINAL UI --- */

    /* 1. BACKGROUND PATTERN (DARK MODE ONLY) - New Abstract Pattern */
    .stApp[data-theme="dark"] {
    background-color: transparent !important;
    background-image: url("https://www.transparenttextures.com/patterns/graphy.png");
    background-repeat: repeat;
    background-size: auto;
}


    /* 2. PLOT CONTAINER (THE TILE) - Now with more transparency */
    /* This now excludes the main tab container by checking that it doesn't contain a header (h2) */
    div[data-testid="stVerticalBlock"]:has(h5):has(.stPlotlyChart):not(:has(h2)) {
        padding: 1rem 1.5rem 1.5rem 1.5rem;
        border-radius: 15px;
        /* The original background-color was rgba(14, 17, 23, 0.85). We reduce the last value (alpha) to increase transparency */
        background-color: rgba(14, 17, 23, 0.7); 
        border: 1px solid #00524C;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 1.5rem;
    }

    /* 3. RESET PLOTLY CHART STYLING */
    div.stPlotlyChart {
        margin-top: 0 !important;
        padding: 0 !important;
        border-radius: 0 !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
    }

    /* Style the title (h5) inside our plot container */
    .plot-container h5 {
        padding-bottom: 10px;
        font-style: italic;
        color: #a0a0a0;
    }

    /* --- 4. CONSOLIDATED MODERN TAB DESIGN --- */

    /* Makes the tab bar 'stick' to the top when scrolling */
    div[data-testid="stTabs"] > div[role="tablist"] {
        position: sticky !important;
        top: 3.2rem; /* Adjust this value if it overlaps with a header */
        z-index: 999;
        background-color: #ffffff; /* Fallback for light mode */
        box-shadow: 0 2px 4px -2px rgba(0,0,0,0.1);
    }
    .stApp[data-theme="dark"] div[data-testid="stTabs"] > div[role="tablist"] {
        background-color: #0e1117 !important; /* Dark mode background */
    }

    /* Styles the text inside the tab button */
    button[data-testid="stTab"] p {
        font-size: 24px !important; /* Large font from previous version */
        font-weight: 600 !important;
        color: #555555;
        transition: color 0.2s ease-in-out;
    }
    .stApp[data-theme="dark"] button[data-testid="stTab"] p {
        color: #a0a0a0;
    }

    /* Styles the tab button container */
    button[data-testid="stTab"] {
        padding: 10px 16px;
        border: none;
        background-color: transparent;
        transition: background-color 0.2s ease-in-out;
        border-radius: 8px 8px 0 0;
    }

    /* Hover effect for inactive tabs */
    button[data-testid="stTab"]:not([aria-selected="true"]):hover {
        background-color: #E6F2E2;
    }
    .stApp[data-theme="dark"] button[data-testid="stTab"]:not([aria-selected="true"]):hover {
        background-color: #00524C;
    }

    /* Style for the currently active tab */
    button[data-testid="stTab"][aria-selected="true"] {
        background-color: #006B54 !important;
    }
    .stApp[data-theme="dark"] button[data-testid="stTab"][aria-selected="true"] {
        background-color: #8CC63F !important;
    }
    button[data-testid="stTab"][aria-selected="true"] p {
        color: white !important;
        font-weight: 700 !important;
    }
    .stApp[data-theme="dark"] button[data-testid="stTab"][aria-selected="true"] p {
        color: #0e1117 !important;
        font-weight: 700 !important;
    }
            
            /* 5. INFO ICON & TOOLTIP */
    .info-icon {
        display: inline-block;
        margin-left: 8px;
        color: #a0a0a0;
        cursor: help;
        position: relative;
    }
    .info-icon .tooltip-text {
        visibility: hidden;
        width: 350px;
        background-color: #262730;
        color: #fff;
        text-align: left;
        font-style: normal;
        font-size: 14px;
        font-weight: normal;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -175px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
    }
    .info-icon:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    /* 6. SECTION NAVIGATION BAR */
    .section-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 10px;
        background-color: rgba(40, 40, 40, 0.5);
        backdrop-filter: blur(5px);
    }
    .section-nav a {
        color: #d0d0d0;
        background-color: #00524C;
        padding: 8px 12px;
        border-radius: 5px;
        text-decoration: none;
        transition: background-color 0.3s, color 0.3s;
    }
    .section-nav a:hover {
        background-color: #8CC63F;
        color: black;
    }

</style>
""", unsafe_allow_html=True)


# ----------------------------------------
# Color Palette
# ----------------------------------------
oldmutual_palette = [
    "#00524C", "#006B54", "#1A8754", "#5AAA46",
    "#8CC63F", "#E6F2E2"
]
negative_color = "#d65f5f"

# ----------------------------------------
# Snowflake Connection & Query Functions
# ----------------------------------------
@st.cache_resource
def get_snowflake_connection():
    """
    Establish and return a Snowflake connection, reading credentials from secrets.toml.
    Using @st.cache_resource to maintain connection across reruns.
    """
    try:
        conn = sf.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"],
            role=st.secrets["snowflake"]["role"],
            client_session_keep_alive=True,
            login_timeout=60,
            network_timeout=60
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        return None

def get_fresh_connection():
    """
    Get a fresh connection for each query to avoid timeout issues
    """
    try:
        conn = sf.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"],
            role=st.secrets["snowflake"]["role"],
            client_session_keep_alive=True,
            login_timeout=60,
            network_timeout=60
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        return None

@st.cache_data(show_spinner=True, ttl=600)  # Cache for 10 minutes
def load_summary_stats():
    """Load basic summary statistics quickly"""
    conn = get_fresh_connection()  # Use fresh connection
    if conn is None:
        return {}

    try:
        sql_columns = f"DESCRIBE TABLE {TABLE_NAME}"
        columns_df = pd.read_sql(sql_columns, conn)
        available_columns = [col.upper() for col in columns_df['name'].tolist()]

        customer_id_col = None
        retail_amount_col = None
        created_at_col = None

        for col in available_columns:
            if 'CUSTOMER' in col and 'ID' in col:
                customer_id_col = col
            if 'RETAIL' in col and 'AMOUNT' in col:
                retail_amount_col = col
            if 'CREATED' in col and 'AT' in col:
                created_at_col = col

        sql = f"""
        SELECT
            COUNT(*) as total_orders,
            {f"COUNT(DISTINCT {customer_id_col}) as unique_customers" if customer_id_col else "0 as unique_customers"},
            {f"SUM({retail_amount_col}) as total_revenue" if retail_amount_col else "0 as total_revenue"},
            {f"AVG({retail_amount_col}) as avg_order_value" if retail_amount_col else "0 as avg_order_value"},
            {f"MIN({created_at_col}) as min_date" if created_at_col else "CURRENT_DATE as min_date"},
            {f"MAX({created_at_col}) as max_date" if created_at_col else "CURRENT_DATE as max_date"}
            FROM {TABLE_NAME}
        """

        result = pd.read_sql(sql, conn)
        result.columns = [col.lower() for col in result.columns]
        return result.iloc[0].to_dict()

    except Exception as e:
        st.error(f"Error loading summary stats: {e}")
        try:
            basic_sql = "SELECT COUNT(*) as total_orders FROM APP_SCHEMA.ORDER_DATA"
            basic_result = pd.read_sql(basic_sql, conn)
            return {
                'total_orders': basic_result.iloc[0]['total_orders'],
                'unique_customers': 0,
                'total_revenue': 0,
                'avg_order_value': 0,
                'min_date': None,
                'max_date': None
            }
        except:
            return {}
    finally:
        if conn:
            conn.close()

@st.cache_data(show_spinner=True, ttl=300)  # Cache for 5 minutes
def load_filtered_data(start_date, end_date, payment_types=None, limit=100000):
    """
    Load filtered data based on user selections with performance optimizations
    """
    conn = get_fresh_connection()
    if conn is None:
        return pd.DataFrame()

    try:
        where_conditions = [
            f"CREATED_AT >= '{start_date}'",
            f"CREATED_AT <= '{end_date}'"
        ]

        if payment_types:
            payment_list = "', '".join(payment_types)
            where_conditions.append(f"PAYMENT_TYPE_NAME IN ('{payment_list}')")

        where_clause = " AND ".join(where_conditions)

        sql = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE {where_clause}
        ORDER BY CREATED_AT DESC
        LIMIT {limit}
        """

        df = pd.read_sql(sql, conn)
        df.columns = [col.lower() for col in df.columns]

        for col in df.columns:
            if col.endswith('_at'):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        numeric_cols = ['birth_year', 'customer_age', 'order_item_qty', 'bundlesize']
        for col in df.columns:
            if not col.endswith('_amount') and not col.endswith('_at') and col not in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)

        numeric_to_convert = ['retail_amount', 'retailer_discounted_amount', 'product_price_fixed',
                              'product_price_min', 'product_price_max', 'order_item_qty',
                              'order_item_retail_amount', 'order_item_provisioned_units',
                              'birth_year', 'customer_age', 'bundlesize']
        for col in numeric_to_convert:
            if col in numeric_to_convert:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'birth_year' in df.columns:
            df['birth_year'] = df['birth_year'].dropna().astype(int)
        if 'customer_age' in df.columns:
            df['customer_age'] = df['customer_age'].dropna().astype(int)

        if 'bundlesize' in df.columns and not df['bundlesize'].isna().all():
            df['bundleSize_MB'] = df['bundlesize'] / (1024 * 1024)

        return df

    except Exception as e:
        st.error(f"Error loading filtered data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=600)
def get_date_range():
    """Get available date range from database"""
    summary = load_summary_stats()
    if summary:
        return summary.get('min_date'), summary.get('max_date')
    return None, None

def validate_data(df):
    """Validate loaded data and show warnings if needed"""
    if df.empty:
        # This is a common case, so st.info is better than st.warning
        return False

    required_cols = ['id', 'created_at', 'retail_amount', 'payment_type_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return False

    if df['retail_amount'].isna().sum() > len(df) * 0.5:
        st.warning("More than 50% of retail_amount values are missing")

    if df['created_at'].isna().sum() > 0:
        st.warning(f"{df['created_at'].isna().sum()} records have missing created_at timestamps")

    return True

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
        # Check if the dataframe contains enough historical data for comparison
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
        # Check for required columns for this specific plot
        if df_tab.empty or 'customer_id_number' not in df_tab.columns:
            st.info("Not enough data to plot weekly KPIs. This chart requires the 'customer_id_number' column.")
            return None
        # Ensure there's more than one week of data to calculate growth
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
        # Cohort analysis needs at least two weeks of data to be meaningful
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
    # 1. Prepare the data
    df = df.sort_values(['sim_msisdn', 'created_at']).copy()
    # Calculate time difference between consecutive purchases for each SIM
    df['time_since_last_purchase'] = df.groupby('sim_msisdn')['created_at'].diff().dt.days
    # Identify the product that preceded this purchase cycle
    df['previous_product'] = df.groupby('sim_msisdn')['product_name'].shift(1)
    
    # We only care about events where a time difference exists (i.e., not the very first purchase)
    df_repurchases = df.dropna(subset=['time_since_last_purchase', 'previous_product'])

    # 2. To keep the chart clean, we'll focus on the Top 7 products that most frequently lead to a repurchase
    top_trigger_products = df_repurchases['previous_product'].value_counts().nlargest(7).index
    
    # 3. Calculate the cumulative distribution for each product segment
    fig = go.Figure()
    # Use a vivid color palette for better line differentiation
    colors = px.colors.qualitative.Vivid 

    for i, product in enumerate(top_trigger_products):
        segment_data = df_repurchases[df_repurchases['previous_product'] == product]
        if segment_data.empty:
            continue
        
        # Calculate the cumulative probability for this product's repurchase cycle
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

    # 4. Style the figure
    fig.update_layout(
        title_text="Repurchase Propensity by Preceding Product",
        xaxis_title="Days Since Last Purchase",
        yaxis_title="Probability of Next Purchase Occurring",
        yaxis_tickformat='.0%',
        height=500,  # Set standard height
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(title="Preceding Product")
    )
    fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.2)')
    
    # Add annotation explaining the chart
    fig.add_annotation(
        text="This shows how quickly customers buy again after purchasing a specific product.",
        align='left',
        showarrow=False,
        xref='paper', yref='paper',
        x=0.02, y=-0.20, # Position below the x-axis label
        font_color="grey"
    )

    return fig

def plot_purchase_sequence_sankey(df):
    """
    Creates a Sankey diagram to visualize the sequence of product purchases
    for the first 4 purchases.
    """
    df = df.sort_values(['sim_msisdn', 'created_at']).copy()

    # 1. Determine the purchase rank for each SIM
    df['purchase_rank'] = df.groupby('sim_msisdn').cumcount() + 1

    # 2. Limit to the first 4 purchases and top 8 products for clarity
    max_rank = 4
    df_sequence = df[df['purchase_rank'] <= max_rank]
    top_products = df_sequence['product_name'].value_counts().nlargest(8).index
    df_sequence['product_name_agg'] = df_sequence['product_name'].apply(lambda x: x if x in top_products else 'Other')

    # 3. Get the next product purchased by the same SIM
    df_sequence['next_product'] = df_sequence.groupby('sim_msisdn')['product_name_agg'].shift(-1)
    df_sequence.dropna(subset=['next_product'], inplace=True) # Remove the last purchase for each SIM

    # 4. Create labels for the Sankey nodes (e.g., "Product A (1st)")
    ordinal_map = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}
    df_sequence['source_label'] = df_sequence['product_name_agg'] + ' (' + df_sequence['purchase_rank'].map(ordinal_map) + ')'
    df_sequence['target_label'] = df_sequence['next_product'] + ' (' + (df_sequence['purchase_rank'] + 1).map(ordinal_map) + ')'

    # 5. Aggregate the paths
    links = df_sequence.groupby(['source_label', 'target_label']).size().reset_index(name='value')
    # For clarity, filter to only show significant paths
    links = links[links['value'] > links['value'].quantile(0.5)]

    # 6. Prepare data for go.Sankey (nodes and links)
    all_nodes = pd.unique(links[['source_label', 'target_label']].values.ravel('K'))
    node_map = {node: i for i, node in enumerate(all_nodes)}

    link_data = dict(
        source=[node_map[src] for src in links['source_label']],
        target=[node_map[tgt] for tgt in links['target_label']],
        value=links['value'],
        color=[oldmutual_palette[s % len(oldmutual_palette)] for s in [node_map[src] for src in links['source_label']]]
    )
    node_data = dict(
        label=all_nodes,
        pad=15,
        thickness=20
    )

    # 7. Create the figure
    fig = go.Figure(go.Sankey(
        node=node_data,
        link=link_data,
        arrangement='snap' # Aligns nodes vertically
    ))

    fig.update_layout(
        title_text="Customer Purchase Journey: Which Product Comes Next?",
        font_size=12,
        height=700,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    return fig

def plot_split_sunbursts(df):
    """
    Creates two side-by-side interactive sunburst charts for revenue analysis:
    1. By Province -> Product Category
    2. By Age Group -> Product Category
    """
    # --- Data Preparation ---
    df_filtered = df[['province', 'age_group', 'product_name', 'retail_amount']].dropna()
    # Create a simpler product category for better visualization
    df_filtered['product_category'] = df_filtered['product_name'].apply(
        lambda x: 'Airtime' if 'airtime' in x.lower() else ('Data' if any(s in x for s in ['MB', 'GB']) else 'Other')
    )

    # --- Create Subplots ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=("Revenue by Province", "Revenue by Age Group")
    )

    # --- Plot 1: Sunburst by Province with theme colors ---
    px_fig_province = px.sunburst(
        df_filtered, path=['province', 'product_category'], values='retail_amount',
        color="product_category",  # <-- Add this line
        color_discrete_sequence=oldmutual_palette
    )
    fig.add_trace(px_fig_province.data[0], row=1, col=1)

    # --- Plot 2: Sunburst by Age Group with theme colors ---
    px_fig_age = px.sunburst(
        df_filtered, path=['age_group', 'product_category'], values='retail_amount',
        color="product_category",  # <-- And add this line
        color_discrete_sequence=oldmutual_palette
    )
    fig.add_trace(px_fig_age.data[0], row=1, col=2)

    # --- Layout and Theming ---
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(t=60, l=10, r=10, b=10)
    )
    # Apply a consistent hover template to both charts
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Total Revenue: R%{value:,.2f}<extra></extra>'
    )

    return fig

def plot_daily_kpis(df):
    """
    Creates a dual-axis line/bar chart of daily revenue and order volume.
    """
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    daily_kpis = df.resample('D', on='created_at').agg(
        total_revenue=('retail_amount', 'sum'),
        order_count=('id', 'count')
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=daily_kpis['created_at'], y=daily_kpis['total_revenue'], name="Total Revenue",
        line=dict(color='#8CC63F'), fill='tozeroy', fillcolor='rgba(140, 198, 63, 0.2)'
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=daily_kpis['created_at'], y=daily_kpis['order_count'], name="Order Count",
        marker_color='#5AAA46', opacity=0.6
    ), secondary_y=True)

    # --- THEME UPDATE ---
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=450  # Set standard height
    )
    fig.update_xaxes(title_text="Date", gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="<b>Total Revenue (R)</b>", secondary_y=False, tickformat="$,.0f", gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="<b>Order Count</b>", secondary_y=True, showgrid=False)
    return fig



def plot_weekly_kpis(df):
    """Plot multiple KPIs on a weekly basis (orders, revenue, customers) using Plotly."""
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Create weekly aggregations, ending the week on Sunday
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

    # --- Row 1: Absolute Values (Line/Area Charts) ---
    fig.add_trace(go.Scatter(
        x=weekly_agg['created_at'], y=weekly_agg['order_count'], name="Orders",
        line=dict(color=oldmutual_palette[0]), fill='tozeroy', mode='lines+markers',
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Orders: %{y:,}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=weekly_agg['created_at'], y=weekly_agg['total_revenue'], name="Revenue",
        line=dict(color=oldmutual_palette[1]), fill='tozeroy', mode='lines+markers',
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Revenue: R%{y:,.2f}<extra></extra>'
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=weekly_agg['created_at'], y=weekly_agg['unique_customers'], name="Customers",
        line=dict(color=oldmutual_palette[2]), fill='tozeroy', mode='lines+markers',
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Customers: %{y:,}<extra></extra>'
    ), row=1, col=3)

    # --- Row 2: Percentage Changes (Bar Charts) ---
    fig.add_trace(go.Bar(
        x=weekly_agg['created_at'], y=weekly_agg['order_growth'], name="Order Growth",
        marker_color=[oldmutual_palette[0] if v >= 0 else negative_color for v in weekly_agg['order_growth'].fillna(0)],
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Growth: %{y:.1%}<extra></extra>',
        opacity=0.6
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=weekly_agg['created_at'], y=weekly_agg['revenue_growth'], name="Revenue Growth",
        marker_color=[oldmutual_palette[1] if v >= 0 else negative_color for v in weekly_agg['revenue_growth'].fillna(0)],
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Growth: %{y:.1%}<extra></extra>',
        opacity=0.6
    ), row=2, col=2)

    fig.add_trace(go.Bar(
        x=weekly_agg['created_at'], y=weekly_agg['customer_growth'], name="Customer Growth",
        marker_color=[oldmutual_palette[2] if v >= 0 else negative_color for v in weekly_agg['customer_growth'].fillna(0)],
        hovertemplate='<b>Week of %{x|%d %b %Y}</b><br>Growth: %{y:.1%}<extra></extra>',
        opacity=0.6
    ), row=2, col=3)

    # --- THEME UPDATE & LAYOUT ---
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
    Creates a stacked bar chart of order volume by product type over time.
    """
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])

    daily_products = df.groupby([df['created_at'].dt.date, 'product_type_desc']).size().reset_index(name='count')
    daily_products.rename(columns={'created_at': 'date'}, inplace=True)

    fig = px.bar(daily_products,
                  x='date',
                  y='count',
                  color='product_type_desc',
                  labels={'date': 'Date', 'count': 'Number of Orders', 'product_type_desc': 'Product Type'},
                  color_discrete_sequence=oldmutual_palette)

    # --- THEME UPDATE ---
    fig.update_layout(
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=450,  # Set standard height
        # Add this legend dictionary to position it at the top
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.2)')
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
        height=500  # Set standard height
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
    # 1. Filter for specific products and define a logical order
    product_order = [
        # Data Bundles (by size)
        '30 MB', '50 MB', '100 MB', '500 MB', '1 GB', '2 GB', '3 GB', '5 GB', '10 GB (30 Days)', '20 GB (30 Days)',
        # WhatsApp Bundles (by duration)
        'WhatsApp Daily', 'WhatsApp Weekly', 'WhatsApp Monthly',
        # Voice Bundles (by duration)
        'Voice 30 Min', 'Voice 100 Min',
        # SMS Bundles (by count)
        '30 SMS', '100 SMS', '500 SMS'
    ]

    df_bundles = df[df['product_name'].isin(product_order)].copy()

    if df_bundles.empty:
        # If no relevant products are in the dataframe, don't create a plot
        st.info("No specific bundle products (e.g., '1 GB', '100 MB') found in the selected data.")
        return None

    df_bundles['product_name'] = pd.Categorical(df_bundles['product_name'], categories=product_order, ordered=True)
    df_bundles.sort_values('product_name', inplace=True)

    # 2. Create the subplot figure (1 row, 2 columns)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
        subplot_titles=("Bundle Orders vs. Revenue", "Weekly Bundle Popularity")
    )

    # --- Plot 1: Bundle Orders vs. Revenue ---

    # Aggregate data for the first plot
    bundle_summary = df_bundles.groupby('product_name', observed=True).agg(
        order_count=('id', 'count'),
        total_revenue=('retail_amount', 'sum')
    ).reset_index()

    # Bar chart for order count
    fig.add_trace(go.Bar(
        x=bundle_summary['product_name'],
        y=bundle_summary['order_count'],
        name='Orders',
        marker_color='#5AAA46',
        hovertemplate='<b>%{x}</b><br>Orders: %{y:,}<extra></extra>'
    ), secondary_y=False, row=1, col=1)

    # Line chart for total revenue
    fig.add_trace(go.Scatter(
        x=bundle_summary['product_name'],
        y=bundle_summary['total_revenue'],
        name='Revenue',
        line=dict(color='#8CC63F', width=3),
        hovertemplate='Revenue: R%{y:,.2f}<extra></extra>'
    ), secondary_y=True, row=1, col=1)

    # --- Plot 2: Weekly Bundle Popularity (Stacked Bar) ---

    df_bundles['created_at'] = pd.to_datetime(df_bundles['created_at'])

    weekly_counts = df_bundles.groupby([
        pd.Grouper(key='created_at', freq='W-SUN'),
        'product_name'
    ], observed=True).size().unstack(fill_value=0)


    # Use a qualitative color scale for the many product categories
    colors = px.colors.qualitative.Vivid

    for i, product in enumerate(weekly_counts.columns):
        fig.add_trace(go.Bar(
            x=weekly_counts.index,
            y=weekly_counts[product],
            name=product,
            marker_color=colors[i % len(colors)],
            hovertemplate=f'<b>{product}</b><br>Week of %{{x|%d %b}}<br>Orders: %{{y}}<extra></extra>'
        ), row=1, col=2)

    # --- Layout and Theming ---
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend_traceorder="normal",
        barmode='stack',  # This is crucial for the second plot
        legend=dict(font=dict(size=10))
    )

    # Style axes for Plot 1
    fig.update_yaxes(title_text="<b>Order Count</b>", secondary_y=False, row=1, col=1, gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="<b>Total Revenue (R)</b>", secondary_y=True, row=1, col=1, showgrid=False)
    fig.update_xaxes(tickangle=45, row=1, col=1)

    # Style axes for Plot 2
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

    # --- Plot 1: Average Order Value & Orders per SIM ---
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
        line=dict(color="#8CC63F", width=3),  # Standardized line color
        hovertemplate='Orders per SIM: %{y:.2f}<extra></extra>'
    ), secondary_y=True, row=1, col=1)

    # --- Plot 2: Product Preference Heatmap ---
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
        colorscale=[oldmutual_palette[0], oldmutual_palette[3]],  # Use palette for gradient
        hovertemplate='<b>Stage:</b> %{y}<br><b>Product:</b> %{x}<br><b>Preference:</b> %{z:.1%}<extra></extra>'
    ), row=1, col=2)

    # --- Plot 3: Cumulative Revenue by Tenure ---
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

    # --- Layout and Theming ---
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
    fig.update_yaxes(title_text="Cumulative Revenue (R)", tickformat="$,.0f", row=2, col=1, gridcolor='rgba(255, 255, 255, 0.2)')

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
        num_colors = len(top_5_cohorts)
        colors = px.colors.sample_colorscale(
            [oldmutual_palette[0], oldmutual_palette[3]],
            [n / (num_colors - 1) for n in range(num_colors)] if num_colors > 1 else [0.5]
        )

        for i, cohort in enumerate(top_5_cohorts):
            cohort_data = revenue_cumulative.loc[cohort]
            fig.add_trace(go.Scatter(
                x=cohort_data.index, y=cohort_data.values,
                name=f"Cohort: {pd.to_datetime(cohort).strftime('%d %b %Y')}",
                line=dict(color=colors[i]),
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

    # --- Layout and Theming ---
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100)  # Add top margin to prevent title/legend overlap
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
        subplot_titles=("Metrics by Province", "Metrics by Age Group")
    )

    # --- Plot 1: Metrics by Province ---
    province_metrics = df.groupby('province').agg(
        total_revenue=('retail_amount', 'sum'),
        order_count=('id', 'count'),
        sim_count=('sim_msisdn', 'nunique')
    ).nlargest(10, 'total_revenue')
    province_metrics['orders_per_sim'] = province_metrics['order_count'] / province_metrics['sim_count']

    fig.add_trace(go.Bar(
        x=province_metrics.index, y=province_metrics['total_revenue'],
        name='Total Revenue', marker_color=oldmutual_palette[2],
        hovertemplate='<b>%{x}</b><br>Revenue: R%{y:,.2f}<extra></extra>',
        opacity=0.6
    ), secondary_y=False, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=province_metrics.index, y=province_metrics['orders_per_sim'],
        name='Orders per SIM', line=dict(color="#8CC63F", width=3),  # Standardized line color
        hovertemplate='Orders per SIM: %{y:.2f}<extra></extra>'
    ), secondary_y=True, row=1, col=1)

    # --- Plot 2: Metrics by Age Group ---
    age_metrics = df.groupby('age_group').agg(
        total_revenue=('retail_amount', 'sum'),
        order_count=('id', 'count'),
        sim_count=('sim_msisdn', 'nunique')
    ).sort_values('total_revenue', ascending=False)
    age_metrics['orders_per_sim'] = age_metrics['order_count'] / age_metrics['sim_count']

    fig.add_trace(go.Bar(
        x=age_metrics.index, y=age_metrics['total_revenue'],
        name='Total Revenue', marker_color=oldmutual_palette[3],
        hovertemplate='<b>%{x}</b><br>Revenue: R%{y:,.2f}<extra></extra>',
        opacity=0.6
    ), secondary_y=False, row=1, col=2)

    fig.add_trace(go.Scatter(
        x=age_metrics.index, y=age_metrics['orders_per_sim'],
        name='Orders per SIM', line=dict(color="#8CC63F", width=3),  # Standardized line color
        hovertemplate='Orders per SIM: %{y:.2f}<extra></extra>'
    ), secondary_y=True, row=1, col=2)

    # --- Layout and Theming ---
    fig.update_layout(
        height=500, showlegend=False,  # Set standard height
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    fig.update_yaxes(title_text="Total Revenue (R)", secondary_y=False, row=1, col=1, gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="Orders per SIM", secondary_y=True, row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="Total Revenue (R)", secondary_y=False, row=1, col=2, gridcolor='rgba(255, 255, 255, 0.2)')
    fig.update_yaxes(title_text="Orders per SIM", secondary_y=True, row=1, col=2, showgrid=False)
    return fig

def plot_correlation_matrix(df):
    """
    Calculates and displays a correlation matrix for key numeric features.
    """
    # Select only relevant numeric columns for correlation
    numeric_df = df[['retail_amount', 'customer_age', 'order_item_retail_amount']].copy()
    numeric_df.rename(columns={
        'retail_amount': 'Order Value',
        'customer_age': 'Customer Age',
        'order_item_retail_amount': 'Order Item Value'
    }, inplace=True)

    correlation_matrix = numeric_df.corr()

    fig = go.Figure(go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale=[[0.0, "#5AAA46"], [1.0, "#006B54"]],
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title_text='Correlation Matrix of Numeric Features',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    return fig

def plot_hierarchical_sunburst(df):
    """
    Creates an interactive sunburst chart to show hierarchical spending patterns.
    """
    # To prevent clutter, limit to top provinces and age groups
    top_provinces = df['province'].value_counts().nlargest(7).index
    df_filtered = df[df['province'].isin(top_provinces)]

    # For the sunburst, we need to handle potential missing values gracefully
    df_filtered = df_filtered[['province', 'age_group', 'product_name', 'retail_amount']].dropna()

    # Create a simpler product category for better visualization
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
    # Get the top 12 most frequent products to keep the matrix readable
    top_12_products = df['product_name'].value_counts().nlargest(12).index
    df_top_products = df[df['product_name'].isin(top_12_products)]

    # Create a matrix where each row is a SIM and each column is a product.
    # The value is 1 if the SIM bought the product, 0 otherwise.
    purchase_matrix = pd.crosstab(df_top_products['sim_msisdn'], df_top_products['product_name'])
    purchase_matrix_binary = (purchase_matrix > 0).astype(int)

    # Calculate the correlation of product purchases
    product_correlation = purchase_matrix_binary.corr()

    fig = go.Figure(go.Heatmap(
        z=product_correlation.values,
        x=product_correlation.columns,
        y=product_correlation.columns,
        colorscale='Greens',
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

# ----------------------------------------
# Helper function for layout
# ----------------------------------------
# ----------------------------------------
# Helper function for layout
# ----------------------------------------
def plot_in_a_box(title: str, fig, info_text: str, anchor_id: str):
    """
    A helper function to group a title, an info icon, a chart, and an HTML anchor.
    """
    if fig is None:
        return

    # Add an invisible HTML anchor for the navigation link
    st.markdown(f"<div id='{anchor_id}'></div>", unsafe_allow_html=True)
    
    with st.container():
        # Display title with a hoverable info icon
        st.markdown(
            f"""
            <h5>
                {title}
                <span class="info-icon">
                    &#8505;
                    <span class="tooltip-text">{info_text}</span>
                </span>
            </h5>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key=title)

# ----------------------------------------
# Function to display content for a single tab
# ----------------------------------------
# ----------------------------------------
# Function to display content for a single tab
# ----------------------------------------
# Info text definitions
info_texts = {
    "kpis": "This chart shows daily total revenue (green line, left axis) and number of orders (green bars, right axis). It helps to spot daily trends and see how sales volume and revenue correlate.",
    "prod_type": "This stacked bar chart displays the daily number of orders, broken down by product type. Each color represents a different product type, allowing you to see which product categories are most popular each day.",
    "weekly_kpis": "This dashboard presents a weekly overview of key performance indicators (KPIs). The top row shows absolute weekly numbers for orders, revenue, and unique customers. The bottom row displays the week-over-week growth rate for each KPI.",
    "propensity": "This chart shows how quickly customers make another purchase after buying a specific top-selling product. A steeper curve means customers repurchase faster. For example, a point at (30, 60%) means 60% of customers who bought that product made their next purchase within 30 days.",
    "growth": "These bars show the week-over-week (WoW) and month-over-month (MoM) growth for revenue and orders, based on the most recent data in the selected range. It's a quick way to assess recent performance momentum.",
    "sankey": "This Sankey diagram illustrates the flow of customers between their first four product purchases. The width of the flow represents the number of customers. This helps to understand common purchase paths and which products lead to follow-up sales.",
    "bundle": "The left chart compares order counts (bars) with revenue (line) for specific product bundles to identify popularity vs profitability. The right chart shows the weekly popularity of these bundles as a stacked bar chart, revealing trends over time.",
    "lifecycle": "This analyzes customer behavior based on how long it has been since their first purchase. It shows average order value, purchase frequency, top product preferences at each stage, and the cumulative revenue generated over their tenure.",
    "cohort": "This provides a cohort analysis based on customer acquisition week. The left plot shows cumulative revenue for the top 5 weekly cohorts. The right plot shows average cumulative revenue per customer, with a projection of the 1-year Lifetime Value (LTV) based on the trend.",
    "geo_demo": "These charts break down revenue and customer engagement by location and age group. Bar charts show total revenue, while line charts show the average number of orders per SIM. This helps identify key customer segments and regions.",
    "prod_corr": "This heatmap shows the correlation between purchases of the top 12 products. A value closer to 1.0 (darker green) indicates that customers who buy one product are more likely to also buy the other, revealing cross-selling opportunities.",
    "sunbursts": "These sunburst charts provide a hierarchical view of revenue. The left chart breaks down revenue by province, then product category. The right chart breaks it down by age group, then product category. Click on segments to drill down."
}

def display_dashboard_content(tab, start_date, end_date):
    """
    Loads data and renders visualizations for a specific tab.
    """
    payment_type_map = {
        "SIM Provisioning": ["SIM_PROVISIONING"],
        "Reward": ["REWARD"],
        "Cost Centre": ["COSTCENTRE"],
        "Standard Payments": ["EXT_VOUCHER", "POS", "VOUCHER", "CUSTOMER_CC"],
        "Airtime": ["AIRTIME"],
    }
    selected_payment_types = payment_type_map.get(tab, [])

    with st.spinner(f"Loading {tab} data..."):
        df = load_filtered_data(start_date, end_date, selected_payment_types, limit=200000)

    if not validate_data(df):
        st.info(f"No {tab} data found for the selected date range.")
        return

    if tab == "Standard Payments":
        # --- Section Definitions for Navigation ---
        sections = {
            "Daily Revenue and Order Volume": "daily-revenue",
            "Order Volume by Product Type": "product-volume",
            "Weekly Performance": "weekly-performance",
            "Repurchase Propensity Analysis": "repurchase-propensity",
            "Recent Growth (WoW & MoM)": "recent-growth",
            "Customer Purchase Journey": "purchase-journey",
            "Bundle Purchase Analysis": "bundle-analysis",
            "Customer Lifecycle Analysis": "customer-lifecycle",
            "Cohort & LTV Analysis": "cohort-analysis",
            "Geographic & Demographic Insights": "geo-insights",
            "Product Purchase Correlation": "product-correlation",
            "Revenue by Province & Age Group": "revenue-breakdown",
        }
        
        # --- Create Navigation Bar ---
        nav_html = "<div class='section-nav'>"
        for title, anchor in sections.items():
            nav_html += f"<a href='#{anchor}'>{title}</a>"
        nav_html += "</div>"
        st.markdown(nav_html, unsafe_allow_html=True)
        
        # --- Pre-generate all figures ---
        fig_kpis = plot_daily_kpis(df)
        fig_prod_type = plot_product_type_timeline(df)
        fig_weekly_kpis = plot_weekly_kpis(df)
        fig_propensity = plot_repurchase_propensity(df)
        fig_growth = plot_growth_bars(df)
        fig_sankey = plot_purchase_sequence_sankey(df)
        fig_bundle = plot_bundle_analysis(df)
        fig_lifecycle = plot_customer_lifecycle(df)
        fig_cohort = plot_cohort_analysis(df)
        fig_geo_demo = plot_geo_demographic_overview(df)
        fig_prod_corr = plot_product_correlation_matrix(df)
        fig_sunbursts = plot_split_sunbursts(df)

        # --- Render Layout using plot_in_a_box ---
        col1, col2 = st.columns(2)
        with col1:
            plot_in_a_box("Daily Revenue and Order Volume", fig_kpis, info_texts["kpis"], "daily-revenue")
        with col2:
            plot_in_a_box("Order Volume by Product Type", fig_prod_type, info_texts["prod_type"], "product-volume")

        plot_in_a_box("Weekly Performance", fig_weekly_kpis, info_texts["weekly_kpis"], "weekly-performance")
        
        col3, col4 = st.columns(2)
        with col3:
            plot_in_a_box("Repurchase Propensity Analysis", fig_propensity, info_texts["propensity"], "repurchase-propensity")
        with col4:
            plot_in_a_box("Recent Growth (WoW & MoM)", fig_growth, info_texts["growth"], "recent-growth")

        plot_in_a_box("Customer Purchase Journey", fig_sankey, info_texts["sankey"], "purchase-journey")
        plot_in_a_box("Bundle Purchase Analysis", fig_bundle, info_texts["bundle"], "bundle-analysis")
        plot_in_a_box("Customer Lifecycle Analysis", fig_lifecycle, info_texts["lifecycle"], "customer-lifecycle")
        plot_in_a_box("Cohort & LTV Analysis", fig_cohort, info_texts["cohort"], "cohort-analysis")
        
        col5, col6 = st.columns(2)
        with col5:
            plot_in_a_box("Geographic & Demographic Insights", fig_geo_demo, info_texts["geo_demo"], "geo-insights")
        with col6:
            plot_in_a_box("Product Purchase Correlation", fig_prod_corr, info_texts["prod_corr"], "product-correlation")

        plot_in_a_box("Revenue by Province & Age Group", fig_sunbursts, info_texts["sunbursts"], "revenue-breakdown")

    elif tab == "Raw":
        st.dataframe(df)
    else:
        # Fallback for other tabs
        st.header(f"{tab} Insights")
        fig_kpis = plot_daily_kpis(df)
        plot_in_a_box(f"Daily Revenue and Orders for {tab}", fig_kpis, "Standard daily KPIs.", "other-kpi")

# ----------------------------------------
# Initialize Application
# ----------------------------------------
summary_stats = load_summary_stats()

# ----------------------------------------
# Sidebar: Title & Filters
# ----------------------------------------
st.sidebar.title("Order Data Filters")
st.sidebar.markdown("Use the controls below to filter data across all tabs.\n")

if summary_stats:
    st.sidebar.metric("Total Orders", f"{summary_stats.get('total_orders', 0):,}")
    st.sidebar.metric("Total Revenue", f"R{summary_stats.get('total_revenue', 0):,.2f}")
    st.sidebar.metric("Unique Customers", f"{summary_stats.get('unique_customers', 0):,}")

min_date, max_date = get_date_range()
if min_date and max_date:
    if isinstance(min_date, pd.Timestamp): min_date = min_date.date()
    elif isinstance(min_date, datetime.datetime): min_date = min_date.date()
    if isinstance(max_date, pd.Timestamp): max_date = max_date.date()
    elif isinstance(max_date, datetime.datetime): max_date = max_date.date()

    default_start_attempt = max_date - datetime.timedelta(days=90)
    default_start = max(default_start_attempt, min_date)

    if min_date == max_date:
        st.sidebar.info(f"Data available for: {min_date}")
    else:
        st.sidebar.info(f"Data available from {min_date} to {max_date}")

    start_date, end_date = st.sidebar.date_input(
        "Date Range:", value=[default_start, max_date], min_value=min_date, max_value=max_date
    )
else:
    st.sidebar.error("Unable to load date range from database")
    start_date, end_date = None, None

# ----------------------------------------
# Main Content Area
# ----------------------------------------
tab_names = ["Standard Payments", "SIM Provisioning", "Reward", "Cost Centre", "Airtime", "Raw"]
tabs = st.tabs(tab_names)


if start_date and end_date:
    # A simple check to ensure we have a valid date range.
    if start_date > end_date:
        st.error("Error: The start date cannot be after the end date. Please select a valid range.")
    else:
        for i, tab_name in enumerate(tab_names):
            with tabs[i]:
                display_dashboard_content(tab_name, start_date, end_date)
else:
    st.info("Select a date range in the sidebar to load and display data.")

# ----------------------------------------
# Footer / Credits
# ----------------------------------------
# 1. Use st.divider() to create a clean horizontal rule.
st.divider()

# 2. Use st.markdown for the text, with unsafe_allow_html=True for the link.
st.markdown(
    '© 2025 FREI. All Rights Reserved. | Support: <a href="mailto:chris@frei.co.za">chris@frei.co.za</a>',
    unsafe_allow_html=True
)

# ----------------------------------------
# Performance Tips (shown in sidebar)
# ----------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Performance Tips")
st.sidebar.markdown("""
- Use shorter date ranges for faster loading
- Data is cached for 10 minutes for summary and 5 minutes for filtered data
- Raw view limited to 25,000 records
""")

if summary_stats:
    st.sidebar.markdown("### Database Overview")
    st.sidebar.markdown(f"""
    - **Total Records**: {summary_stats.get('total_orders', 0):,}
    - **Date Range**: {summary_stats.get('min_date', 'N/A')} to {summary_stats.get('max_date', 'N/A')}
    - **Avg Order Value**: R{summary_stats.get('avg_order_value', 0):.2f}
    """)