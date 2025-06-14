import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Order Data Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /*
    MODERN TAB DESIGN by Gemini (v4)
    - NEW: Filled style for active tab with white font.
    - FIX: More specific selectors for font styling to override defaults.
    */

    /* --- GENERAL TAB STYLES --- */

    /* This targets the text element INSIDE the button for all font styling */
    button[data-testid="stTab"] p {
        font-size: 24px !important;  /* <-- INCREASED FONT SIZE (adjust as needed) */
        font-weight: 600 !important;
        color: #555555;
        transition: color 0.2s ease-in-out;
    }

    /* Dark mode text color for inactive tabs */
    .stApp[data-theme="dark"] button[data-testid="stTab"] p {
        color: #a0a0a0;
    }

    /* This targets the button container itself for shape and spacing */
    button[data-testid="stTab"] {
        padding: 10px 16px;
        border: none;
        background-color: transparent;
        transition: background-color 0.2s ease-in-out;
        border-radius: 8px 8px 0 0; /* Rounded top corners for a modern look */
    }

    /* Hover effect for inactive tabs */
    button[data-testid="stTab"]:not([aria-selected="true"]):hover {
        background-color: #E6F2E2;
    }
    .stApp[data-theme="dark"] button[data-testid="stTab"]:not([aria-selected="true"]):hover {
        background-color: #00524C;
    }


    /* --- ACTIVE TAB STYLE (FILLED) --- */

    /* This targets the active button container to give it a background color */
    button[data-testid="stTab"][aria-selected="true"] {
        background-color: #006B54 !important; /* Main theme color as background */
    }
    .stApp[data-theme="dark"] button[data-testid="stTab"][aria-selected="true"] {
        background-color: #8CC63F !important; /* Bright green as background in dark mode */
    }

    /* This targets the text inside the active button to make it white */
    button[data-testid="stTab"][aria-selected="true"] p {
        color: white !important; /* <-- WHITE FONT COLOR FOR ACTIVE TAB */
        font-weight: 700 !important;
    }
    /* In dark mode, we use a dark font on the bright green background for contrast */
    .stApp[data-theme="dark"] button[data-testid="stTab"][aria-selected="true"] p {
        color: #0e1117 !important; 
        font-weight: 700 !important;
    }

</style>
""", unsafe_allow_html=True)


# ----------------------------------------
# Color Palette
# ----------------------------------------
oldmutual_palette = [
    "#006B54", "#1A8754", "#5AAA46",
    "#8CC63F", "#00524C", "#E6F2E2"
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
        # First, let's get the table structure to see what columns exist
        # sql_columns = "DESCRIBE TABLE APP_SCHEMA.ORDER_DATA"
        sql_columns = f"DESCRIBE TABLE {TABLE_NAME}"
        columns_df = pd.read_sql(sql_columns, conn)
        available_columns = [col.upper() for col in columns_df['name'].tolist()]

        # Build a flexible query based on available columns
        customer_id_col = None
        retail_amount_col = None
        created_at_col = None

        # Find the right column names
        for col in available_columns:
            if 'CUSTOMER' in col and 'ID' in col:
                customer_id_col = col
            if 'RETAIL' in col and 'AMOUNT' in col:
                retail_amount_col = col
            if 'CREATED' in col and 'AT' in col:
                created_at_col = col

        # Build the query with available columns
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
        # Convert column names to lowercase for consistency
        result.columns = [col.lower() for col in result.columns]
        return result.iloc[0].to_dict()

    except Exception as e:
        st.error(f"Error loading summary stats: {e}")
        # Return basic stats if query fails
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
@st.cache_data(show_spinner=True, ttl=300)  # Cache for 5 minutes
def load_filtered_data(start_date, end_date, payment_types=None, limit=50000):
    """
    Load filtered data based on user selections with performance optimizations
    """
    conn = get_fresh_connection()  # Use fresh connection
    if conn is None:
        return pd.DataFrame()

    try:
        # Build WHERE clause with uppercase column names (Snowflake style)
        where_conditions = [
            f"CREATED_AT >= '{start_date}'",
            f"CREATED_AT <= '{end_date}'"
        ]

        if payment_types:
            payment_list = "', '".join(payment_types)
            where_conditions.append(f"PAYMENT_TYPE_NAME IN ('{payment_list}')")

        where_clause = " AND ".join(where_conditions)

        # Simple approach - select all columns and let pandas handle it
        sql = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE {where_clause}
        ORDER BY CREATED_AT DESC
        LIMIT {limit}
        """

        df = pd.read_sql(sql, conn)

        # Data type conversions - convert all column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # --- START: INSERT YOUR NEW CODE HERE ---

        # Convert all columns ending with '_at' to datetime, coercing errors
        for col in df.columns:
            if col.endswith('_at'):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # If column in df does not contain a '_amount' change the type to string
        # Exclude other specific numeric columns
        numeric_cols = ['birth_year', 'customer_age', 'order_item_qty', 'bundlesize'] # Add any other known numerics
        for col in df.columns:
            if not col.endswith('_amount') and not col.endswith('_at') and col not in numeric_cols:
                if col in df.columns: # Check if column exists before conversion
                    df[col] = df[col].astype(str)

        # Convert specific columns to numeric types safely
        # Using pd.to_numeric is safer as it handles potential non-numeric values gracefully
        numeric_to_convert = ['retail_amount', 'retailer_discounted_amount', 'product_price_fixed',
                              'product_price_min', 'product_price_max', 'order_item_qty',
                              'order_item_retail_amount', 'order_item_provisioned_units',
                              'birth_year', 'customer_age', 'bundlesize']
        for col in numeric_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert birth_year and age to int after handling decimals/errors
        if 'birth_year' in df.columns:
            df['birth_year'] = df['birth_year'].dropna().astype(int)
        if 'customer_age' in df.columns:
            df['customer_age'] = df['customer_age'].dropna().astype(int)

        # --- END: NEW CODE SECTION ---

        # Create bundleSize_MB if bundlesize exists
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
        st.warning("No data found for the selected criteria. Try expanding your date range or changing filters.")
        return False

    # Check for required columns
    required_cols = ['id', 'created_at', 'retail_amount', 'payment_type_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return False

    # Check data quality
    if df['retail_amount'].isna().sum() > len(df) * 0.5:
        st.warning("More than 50% of retail_amount values are missing")

    if df['created_at'].isna().sum() > 0:
        st.warning(f"{df['created_at'].isna().sum()} records have missing created_at timestamps")

    return True

# ----------------------------------------
# Plotting Functions with Error Handling
# ----------------------------------------
def safe_plot_weekly_kpis(df_tab):
    """Wrapper for plot_weekly_kpis_plotly with error handling"""
    try:
        if df_tab.empty:
            st.info("No data available for weekly KPI analysis")
            return None

        if len(df_tab) < 7:  # Less than a week of data
            st.warning("Insufficient data for meaningful weekly analysis")
            return None

        return plot_weekly_kpis_plotly(df_tab)
    except Exception as e:
        st.error(f"Error creating weekly KPI chart: {e}")
        return None

def safe_plot_bundle_analysis(df_tab):
    """Wrapper for bundle analysis with data checks"""
    try:
        if df_tab.empty:
            st.info("No data available for bundle analysis")
            return None

        # Check if we have bundle size data
        has_bundle_data = (
            'bundlesize' in df_tab.columns and
            not df_tab['bundlesize'].isna().all()
        ) or (
            'bundleSize_MB' in df_tab.columns and
            not df_tab['bundleSize_MB'].isna().all()
        )

        if not has_bundle_data:
            st.warning("No bundle size data available for this selection")
            return None

        return plot_bundle_size_analysis(df_tab)
    except Exception as e:
        st.error(f"Error creating bundle analysis: {e}")
        return None

def safe_plot_temporal_patterns(df_tab):
    """Wrapper for temporal patterns with error handling"""
    try:
        if df_tab.empty:
            st.info("No data available for temporal analysis")
            return None
        return plot_temporal_patterns(df_tab)
    except Exception as e:
        st.error(f"Error creating temporal patterns chart: {e}")
        return None

def safe_plot_geographic_analysis(df_tab):
    """Wrapper for geographic analysis with error handling"""
    try:
        if df_tab.empty:
            st.info("No data available for geographic analysis")
            return None
        return plot_geographic_analysis(df_tab)
    except Exception as e:
        st.error(f"Error creating geographic analysis: {e}")
        return None

def safe_plot_customer_lifecycle_value(df_tab):
    """Wrapper for customer lifecycle analysis with error handling"""
    try:
        if df_tab.empty:
            st.info("No data available for customer lifecycle analysis")
            return None
        return plot_customer_lifecycle_value(df_tab)
    except Exception as e:
        st.error(f"Error creating customer lifecycle analysis: {e}")
        return None

def safe_plot_customer_lifetime_cohort(df_tab):
    """Wrapper for cohort analysis with error handling"""
    try:
        if df_tab.empty:
            st.info("No data available for cohort analysis")
            return None
        return plot_customer_lifetime_cohort(df_tab)
    except Exception as e:
        st.error(f"Error creating cohort analysis: {e}")
        return None

# ----------------------------------------
# Plotly Visualization Functions
# ----------------------------------------
def plot_weekly_kpis_plotly(df_tab):
    # Resample weekly on Sunday
    weekly_orders = df_tab.resample('W-SUN', on='created_at').size()
    weekly_revenue = df_tab.resample('W-SUN', on='created_at')['retail_amount'].sum()
    weekly_customers = df_tab.resample('W-SUN', on='created_at')['customer_id_number'].nunique()

    # Percent changes
    weekly_orders_pct = weekly_orders.pct_change() * 100
    weekly_revenue_pct = weekly_revenue.pct_change() * 100
    weekly_customers_pct = weekly_customers.pct_change() * 100

    # Create subplots: 2 rows x 3 cols
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Weekly Order Volume",
            "Weekly Revenue",
            "Weekly Unique Customers",
            "Weekly Order Growth (%)",
            "Weekly Revenue Growth (%)",
            "Weekly Customer Growth (%)"
        ),
        horizontal_spacing=0.05,
        vertical_spacing=0.15
    )

    # Row 1, Col 1: Weekly Orders (Line + Fill)
    fig.add_trace(
        go.Scatter(
            x=weekly_orders.index,
            y=weekly_orders.values,
            mode="lines+markers",
            line=dict(color=oldmutual_palette[0], width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 107, 84, 0.2)",  # 20% opacity
            name="Orders"
        ),
        row=1, col=1
    )

    # Row 1, Col 2: Weekly Revenue
    fig.add_trace(
        go.Scatter(
            x=weekly_revenue.index,
            y=weekly_revenue.values,
            mode="lines+markers",
            line=dict(color=oldmutual_palette[1], width=2),
            fill="tozeroy",
            fillcolor="rgba(26, 135, 84, 0.2)",
            name="Revenue"
        ),
        row=1, col=2
    )

    # Row 1, Col 3: Weekly Unique Customers
    fig.add_trace(
        go.Scatter(
            x=weekly_customers.index,
            y=weekly_customers.values,
            mode="lines+markers",
            line=dict(color=oldmutual_palette[2], width=2),
            fill="tozeroy",
            fillcolor="rgba(90, 170, 70, 0.2)",
            name="Customers"
        ),
        row=1, col=3
    )

    # Row 2, Col 1: Weekly Order % Change (Bar)
    fig.add_trace(
        go.Bar(
            x=weekly_orders_pct.index[1:],
            y=weekly_orders_pct.values[1:],
            marker_color=[
                oldmutual_palette[0] if val >= 0 else negative_color
                for val in weekly_orders_pct.values[1:]
            ],
            name="% Orders"
        ),
        row=2, col=1
    )

    # Row 2, Col 2: Weekly Revenue % Change (Bar)
    fig.add_trace(
        go.Bar(
            x=weekly_revenue_pct.index[1:],
            y=weekly_revenue_pct.values[1:],
            marker_color=[
                oldmutual_palette[1] if val >= 0 else negative_color
                for val in weekly_revenue_pct.values[1:]
            ],
            name="% Revenue"
        ),
        row=2, col=2
    )

    # Row 2, Col 3: Weekly Customers % Change (Bar)
    fig.add_trace(
        go.Bar(
            x=weekly_customers_pct.index[1:],
            y=weekly_customers_pct.values[1:],
            marker_color=[
                oldmutual_palette[2] if val >= 0 else negative_color
                for val in weekly_customers_pct.values[1:]
            ],
            name="% Customers"
        ),
        row=2, col=3
    )

    # Update all subplots' layout
    for i in range(1, 7):
        row = 1 if i <= 3 else 2
        col = i if i <= 3 else i - 3
        # Format axes
        fig.update_xaxes(
            row=row, col=col,
            tickformat="%b %d",
            tickangle=45,
            nticks=10
        )
        if row == 1:
            if col == 2:
                # Currency formatting
                fig.update_yaxes(
                    row=row, col=col,
                    tickformat="$,",
                    title_text="Revenue"
                )
            else:
                # Integer formatting
                fig.update_yaxes(
                    row=row, col=col,
                    tickformat=",",
                    title_text=(
                        "Number of Orders" if col == 1
                        else "Number of Customers"
                    )
                )
        else:
            # Percentage formatting
            fig.update_yaxes(
                row=row, col=col,
                tickformat=",.1f%",
                title_text="% Change"
            )

    fig.update_layout(
        height=700,
        showlegend=False,
        title_font_size=16,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig

def plot_bundle_size_analysis(df):
    """
    Create comprehensive analysis of data bundle sizes and their popularity, using Plotly.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # 1) Ensure bundleSize_MB exists
    if 'bundleSize_MB' not in df.columns and 'bundlesize' in df.columns:
        df['bundleSize_MB'] = df['bundlesize'] / (1024 * 1024)

    # 2) Filter out invalid or zero bundle-size rows
    bundle_data = df.dropna(subset=['bundleSize_MB'])
    bundle_data = bundle_data[bundle_data['bundleSize_MB'] > 0]

    if bundle_data.empty:
        st.warning("No valid bundle size data available")
        return None

    # 3) Define common bins & labels
    common_sizes = [30, 50, 100, 250, 500, 1024, 2048, 5120, 10240, 20480]
    size_labels   = ['30MB', '50MB', '100MB', '250MB', '500MB', '1GB', '2GB', '5GB', '10GB', '20GB']

    bundle_data['size_category'] = pd.cut(
        bundle_data['bundleSize_MB'],
        bins=[0] + common_sizes + [float('inf')],
        labels=['<30MB'] + size_labels
    )

    # Grab the category order as defined
    all_categories = bundle_data['size_category'].cat.categories.tolist()

    # Distribution counts
    counts = (
        bundle_data['size_category']
        .value_counts()
        .reindex(all_categories, fill_value=0)
    )

    # Create a simple bar chart for now
    fig = go.Figure(go.Bar(
        x=all_categories,
        y=counts.values,
        marker_color=oldmutual_palette[0],
        text=[f"{int(v):,}" for v in counts.values],
        textposition='outside',
        name='Count'
    ))

    fig.update_layout(
        title="Data Bundle Size Distribution",
        xaxis_title="Bundle Size Category",
        yaxis_title="Number of Orders",
        height=500
    )

    return fig

def plot_temporal_patterns(df):
    """
    Create Plotly visualizations of temporal patterns and seasonality in the data.
    """
    df = df.copy()

    # Weekly metrics (resample by week ending Sunday)
    df['week'] = df['created_at'].dt.to_period('W-SUN').dt.to_timestamp()
    weekly_orders = df.groupby('week').size().rename('orders')
    weekly_revenue = df.groupby('week')['retail_amount'].sum().rename('revenue')

    # Day-of-week metrics
    df['day_of_week'] = df['created_at'].dt.day_name()
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_metrics = (
        df.groupby('day_of_week')
          .agg(
              orders=('id','count'),
              revenue=('retail_amount','sum')
          )
          .reindex(day_order)
          .reset_index()
    )

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Weekly Orders & Revenue",
            "Orders by Day of Week",
            "Weekly Revenue",
            "Revenue by Day of Week"
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )

    # Weekly Orders & Revenue
    fig.add_trace(
        go.Bar(
            x=weekly_orders.index,
            y=weekly_orders.values,
            marker_color=oldmutual_palette[0],
            name="Orders",
            opacity=0.7
        ),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=weekly_revenue.index,
            y=weekly_revenue.values,
            mode='lines+markers',
            line=dict(color=oldmutual_palette[1], width=2),
            name="Revenue",
        ),
        row=1, col=1, secondary_y=True
    )

    # Orders by Day of Week
    fig.add_trace(
        go.Bar(
            x=day_metrics['day_of_week'],
            y=day_metrics['orders'],
            marker_color=oldmutual_palette[2],
            name="Orders by Day",
        ),
        row=1, col=2
    )

    # Weekly Revenue
    fig.add_trace(
        go.Scatter(
            x=weekly_revenue.index,
            y=weekly_revenue.values,
            mode='lines+markers',
            line=dict(color=oldmutual_palette[1], width=2),
            fill='tonexty',
            name="Weekly Revenue",
        ),
        row=2, col=1
    )

    # Revenue by Day of Week
    fig.add_trace(
        go.Bar(
            x=day_metrics['day_of_week'],
            y=day_metrics['revenue'],
            marker_color=oldmutual_palette[3],
            name="Revenue by Day",
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=60)
    )

    return fig

def plot_geographic_analysis(df):
    """
    Create Plotly visualizations of geographic patterns using derived 'province' data.
    """
    if 'customer_address5' not in df.columns:
        st.warning("'customer_address5' column not found in dataframe")
        return None

    df_copy = df.copy()

    # Simple geographic analysis based on available data
    if 'province' in df_copy.columns:
        province_data = df_copy['province'].value_counts().head(10)

        fig = go.Figure(go.Bar(
            x=province_data.values,
            y=province_data.index,
            orientation='h',
            marker_color=oldmutual_palette[0]
        ))

        fig.update_layout(
            title="Orders by Province",
            xaxis_title="Number of Orders",
            yaxis_title="Province",
            height=400
        )

        return fig
    else:
        st.info("Province data not available for geographic analysis")
        return None

def plot_customer_lifecycle_value(df):
    """
    Create visualizations showing customer behavior and value across their lifecycle.
    """
    df = df.copy()

    # Calculate days since acquisition
    df['acquisition_date'] = df.groupby('customer_id_number')['customer_created_at'].transform('min')
    df['days_since_acquisition'] = (df['created_at'] - df['acquisition_date']).dt.days

    # Create lifecycle stage categories
    df['lifecycle_stage'] = pd.cut(
        df['days_since_acquisition'],
        bins=[-1, 7, 30, 90, 180, 365, float('inf')],
        labels=['First Week', '8-30 Days', '31-90 Days', '91-180 Days', '181-365 Days', '365+ Days']
    )

    # Compute metrics by stage
    stage_metrics = (
        df.groupby('lifecycle_stage')
          .agg(
              avg_order_value=('retail_amount', 'mean'),
              num_orders=('id', 'count'),
              num_customers=('customer_id_number', pd.Series.nunique)
          )
          .reset_index()
    )

    # Create bar chart
    fig = go.Figure(go.Bar(
        x=stage_metrics['lifecycle_stage'].astype(str),
        y=stage_metrics['avg_order_value'],
        marker_color=oldmutual_palette[0],
        text=[f"R{v:.2f}" for v in stage_metrics['avg_order_value']],
        textposition='outside'
    ))

    fig.update_layout(
        title="Average Order Value by Lifecycle Stage",
        xaxis_title="Lifecycle Stage",
        yaxis_title="Average Order Value (R)",
        height=400
    )

    return fig

def plot_customer_lifetime_cohort(df):
    """
    Create simplified cohort analysis visualization.
    """
    df = df.copy()

    def get_month_year(dt):
        return pd.Period(dt, freq='M') if not pd.isna(dt) else None

    df['cohort_month'] = (
        df.groupby('customer_id_number')['customer_created_at']
          .transform('min')
          .apply(get_month_year)
    )
    df['activity_month'] = df['created_at'].apply(get_month_year)

    df_cohort = df.dropna(subset=['cohort_month', 'activity_month']).copy()
    df_cohort['cohort_month_num'] = (
        df_cohort['activity_month'].astype(int) - df_cohort['cohort_month'].astype(int)
    )
    df_cohort = df_cohort[df_cohort['cohort_month_num'] >= 0]

    # Simple cohort retention for first 6 months
    cohort_data = (
        df_cohort[df_cohort['cohort_month_num'] <= 6]
        .groupby(['cohort_month', 'cohort_month_num'])['customer_id_number']
        .nunique()
        .reset_index(name='unique_customers')
    )

    if cohort_data.empty:
        st.info("Insufficient data for cohort analysis")
        return None

    cohort_pivot = cohort_data.pivot(
        index='cohort_month', columns='cohort_month_num', values='unique_customers'
    ).fillna(0)

    if cohort_pivot.empty:
        st.info("No cohort data available")
        return None

    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_rates = cohort_pivot.divide(cohort_sizes, axis=0).fillna(0)

    # Create heatmap
    fig = go.Figure(go.Heatmap(
        z=retention_rates.values,
        x=retention_rates.columns.tolist(),
        y=[str(p) for p in retention_rates.index.tolist()],
        colorscale=oldmutual_palette,
        hovertemplate="Cohort: %{y}<br>Month %{x}: %{z:.0%}<extra></extra>",
        colorbar=dict(title="% Retention")
    ))

    fig.update_layout(
        title="Monthly Cohort Retention Rate",
        xaxis_title="Months Since Acquisition",
        yaxis_title="Acquisition Cohort",
        height=500
    )

    return fig


# ----------------------------------------
# Function to display content for a single tab
# ----------------------------------------
def display_dashboard_content(tab, start_date, end_date):
    """
    Loads data and renders all visualizations for a specific tab.
    """
    # Payment-Type Mapping
    payment_type_map = {
        "SIM Provisioning": ["SIM_PROVISIONING"],
        "Reward": ["REWARD"],
        "Cost Centre": ["COSTCENTRE"],
        "Standard Payments": ["EXT_VOUCHER", "POS", "VOUCHER", "CUSTOMER_CC"],
        "Airtime": ["AIRTIME"],
    }

    # Get payment types for selected tab
    selected_payment_types = payment_type_map.get(tab, [])

    # Load data with progress indicator
    with st.spinner(f"Loading {tab} data..."):
        if tab == "Raw":
            df = load_filtered_data(start_date, end_date, limit=25000)
        else:
            df = load_filtered_data(start_date, end_date, selected_payment_types)

    # Filter data for the selected payment types (additional client-side filter)
    if tab != "Raw" and not df.empty:
        df = df[df["payment_type_name"].isin(selected_payment_types)].copy()

    # Validate data
    if not df.empty:
        data_valid = validate_data(df)
        if not data_valid:
            st.stop()

    # Display Content Based on Tab Selection
    if tab == "Raw":
        st.header("Raw Data Table")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data to display for the selected date range.")
    else:
        st.header(f"{tab} Insights")

        if df.empty:
            st.info(f"No {tab} data found for the selected date range.")
        else:
            count = len(df)
            st.subheader(f"Data Points: {count:,}")
            st.write(f"Visualizations for **{tab}**, between **{start_date}** and **{end_date}**.")

            # Debug information - show what data we actually have
            if not df.empty:
                st.markdown("### Data Debug Information")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Rows", f"{len(df):,}")

                with col2:
                    if 'created_at' in df.columns:
                        date_range = (df['created_at'].max() - df['created_at'].min()).days
                        st.metric("Date Range (Days)", f"{date_range}")
                    else:
                        st.metric("Date Range", "No date column")

                with col3:
                    if 'created_at' in df.columns:
                        unique_dates = df['created_at'].dt.date.nunique()
                        st.metric("Unique Dates", f"{unique_dates}")
                    else:
                        st.metric("Unique Dates", "N/A")

                # Show date range
                if 'created_at' in df.columns:
                    st.write(f"**Date Range in Data:** {df['created_at'].min()} to {df['created_at'].max()}")

                # Show sample data
                st.markdown("**Sample Data:**")
                display_cols = ['created_at', 'retail_amount', 'product_name', 'payment_type_name']
                available_cols = [col for col in display_cols if col in df.columns]
                st.dataframe(df[available_cols].head(), use_container_width=True)

            # 1) Weekly KPIs
            st.markdown("### Weekly KPI Trends")
            st.markdown("""
            **Description**:
            This chart panel shows six subplots:
            1. **Weekly Order Volume** (lines+markers): total number of orders each week (resampled to Sundays).
            2. **Weekly Revenue** (lines+markers): sum of `retail_amount` each week.
            3. **Weekly Unique Customers** (lines+markers): count of distinct `customer_id_number` each week.
            4. **Weekly Order Growth (%)** (bars): percent change in order volume from the previous week.
            5. **Weekly Revenue Growth (%)** (bars): percent change in revenue from the previous week.
            6. **Weekly Customer Growth (%)** (bars): percent change in unique customers from the previous week.
            """)

            fig1 = safe_plot_weekly_kpis(df)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)

            # 2) Bundle Size Distribution (only for Standard Payments)
            if tab == "Standard Payments":
                st.markdown("### Bundle Size Distribution Analysis")
                st.markdown("""
                **Description**:
                A histogram showing how many orders fall into each *data bundle size* category (e.g., `<30MB`, `30MB`, `50MB`, â€¦, `20GB`). Each bar's height represents the count of orders in that size range.
                """)

                fig2 = safe_plot_bundle_analysis(df)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### Customer Lifecycle Value Analysis")
                st.markdown("""
                **Description**:
                Shows how customer-related metrics evolve by "lifecycle stage" (based on days since their first purchase). Includes average order value by lifecycle stage.
                """)

                fig3 = safe_plot_customer_lifecycle_value(df)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Bundle size and lifecycle analyses are only applicable for Standard Payments.")

            # 3) Temporal Patterns & Seasonality Analysis
            st.markdown("### Temporal Patterns & Seasonality Analysis")
            st.markdown("""
            **Description**:
            This panel visualizes time-based order behavior including weekly trends and day-of-week patterns.
            """)

            fig4 = safe_plot_temporal_patterns(df)
            if fig4:
                st.plotly_chart(fig4, use_container_width=True)

            # 4) Geographic Analysis
            st.markdown("### Geographic Analysis")
            st.markdown("""
            **Description**:
            Shows how order metrics vary by geographic region based on available location data.
            """)

            fig5 = safe_plot_geographic_analysis(df)
            if fig5:
                st.plotly_chart(fig5, use_container_width=True)

            # 5) Cohort Analysis
            st.markdown("### Cohort Analysis")
            st.markdown("""
            **Description**:
            This chart displays cohort-based retention analysis showing the percentage of customers remaining active in subsequent months.
            """)

            fig6 = safe_plot_customer_lifetime_cohort(df)
            if fig6:
                st.plotly_chart(fig6, use_container_width=True)

            # 6) Filtered Data Table
            st.markdown("----")
            st.subheader("Filtered Data Table")

            # Show only first 1000 rows for performance
            display_df = df.head(1000)
            st.dataframe(display_df, use_container_width=True)

            if len(df) > 1000:
                st.info(f"Showing first 1,000 rows of {len(df):,} total records")

            # 7) Basic Statistics & Data Quality Diagnostics
            st.markdown("### â–¶ï¸ Basic Statistics & Data Quality Diagnostics")

            # Numeric columns: describe()
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if num_cols:
                with st.expander("Numeric columns summary"):
                    st.dataframe(df[num_cols].describe())

            # Missing values analysis
            missing_counts = df.isna().sum()
            if missing_counts.sum() > 0:
                missing_pct = (missing_counts / len(df) * 100).round(2)
                miss_df = pd.concat([missing_counts, missing_pct], axis=1, keys=["missing_count", "missing_pct"])
                miss_df = miss_df[miss_df["missing_count"] > 0]

                with st.expander("Missing value analysis"):
                    st.dataframe(miss_df)
            else:
                st.success("âœ“ No missing values detected")

            # Duplicate rows analysis
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                st.warning(f"âš ï¸ Found {dup_count} duplicate rows")
            else:
                st.success("âœ“ No duplicate rows found")

            # Data quality summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Completeness", f"{((1 - missing_counts.sum() / (len(df) * len(df.columns))) * 100):.1f}%")
            with col2:
                st.metric("Unique Customers", f"{df['customer_id_number'].nunique():,}")
            with col3:
                st.metric("Date Range", f"{(df['created_at'].max() - df['created_at'].min()).days} days")


# ----------------------------------------
# Initialize Application
# ----------------------------------------

# Load summary statistics first (fast)
summary_stats = load_summary_stats()

# ----------------------------------------
# Sidebar: Title & Filters
# ----------------------------------------
st.sidebar.title("Order Data Filters")
st.sidebar.markdown("Use the controls below to filter data across all tabs.\n")

# Display summary stats in sidebar
if summary_stats:
    st.sidebar.metric("Total Orders", f"{summary_stats.get('total_orders', 0):,}")
    st.sidebar.metric("Total Revenue", f"R{summary_stats.get('total_revenue', 0):,.2f}")
    st.sidebar.metric("Unique Customers", f"{summary_stats.get('unique_customers', 0):,}")


# Date Range Filter with dynamic defaults
min_date, max_date = get_date_range()
if min_date and max_date:
    # Ensure we have date objects, not datetime objects
    import datetime

    if isinstance(min_date, pd.Timestamp):
        min_date = min_date.date()
    elif isinstance(min_date, datetime.datetime):
        min_date = min_date.date()

    if isinstance(max_date, pd.Timestamp):
        max_date = max_date.date()
    elif isinstance(max_date, datetime.datetime):
        max_date = max_date.date()

    # Calculate default start date, but ensure it's not before min_date
    default_start_attempt = max_date - datetime.timedelta(days=90)
    default_start = max(default_start_attempt, min_date)  # Use the later of the two dates

    # Show available date range to user
    if min_date == max_date:
        st.sidebar.info(f"“… Data available for: {min_date}")
    else:
        st.sidebar.info(f"“… Data available from {min_date} to {max_date}")

    start_date, end_date = st.sidebar.date_input(
        "Date Range:",
        value=[default_start, max_date],
        min_value=min_date,
        max_value=max_date,
    )
else:
    st.sidebar.error("Unable to load date range from database")
    start_date, end_date = None, None

# ----------------------------------------
# Main Content Area
# ----------------------------------------
# ----------------------------------------
# Create tabs and render content within each
# ----------------------------------------
tab_names = ["Standard Payments", "SIM Provisioning", "Reward", "Cost Centre", "Airtime", "Raw"]
tabs = st.tabs(tab_names)

# ----------------------------------------
# Modern Custom CSS for Tabs
# ----------------------------------------
st.markdown("""
<style>
    /*
    MODERN TAB DESIGN by Gemini
    This CSS provides a clean, modern, and theme-aware tab design.
    */

    /* Make the tab bar sticky */
    div[data-testid="stTabs"] > div[role="tablist"] {
        position: sticky !important;
        top: 3.2rem; /* Adjust this to prevent overlap with your header */
        z-index: 999;
        box-shadow: 0 2px 4px -2px rgba(0,0,0,0.1);
        border-bottom: 2px solid #E6F2E2; /* Light green separator line */
    }

    /* Style for the tab bar in Dark Theme */
    .stApp[data-theme="dark"] div[data-testid="stTabs"] > div[role="tablist"] {
        background-color: #0e1117 !important; /* Streamlit's default dark background */
        border-bottom: 2px solid #00524C; /* Dark green separator for dark mode */
    }

    /* Style for individual tab buttons */
    button[data-testid="stTab"] {
        padding: 12px 18px;
        color: #444444; /* Darker grey for better readability */
        background-color: transparent;
        border: none;
        border-bottom: 3px solid transparent; /* Prepare for the active indicator line */
        transition: all 0.2s ease-in-out;
        font-weight: 500;
        font-size: 15px;
    }

    /* Style for tab buttons in Dark Theme */
    .stApp[data-theme="dark"] button[data-testid="stTab"] {
        color: #a0a0a0;
    }

    /* Hover effect for tab buttons */
    button[data-testid="stTab"]:hover {
        background-color: #E6F2E2; /* Lightest green from palette for hover */
        color: #006B54; /* Main green for text on hover */
        border-bottom: 3px solid #5AAA46; /* A medium green for the bottom border on hover */
    }

    /* Hover effect for tab buttons in Dark Theme */
    .stApp[data-theme="dark"] button[data-testid="stTab"]:hover {
        background-color: #00524C; /* Dark green for hover background in dark mode */
        color: #ffffff;
        border-bottom: 3px solid #1A8754;
    }

    /* Style for the ACTIVE tab button */
    button[data-testid="stTab"][aria-selected="true"] {
        color: #006B54; /* Main theme color for active tab text */
        font-weight: 700;
        border-bottom: 3px solid #006B54; /* Main theme color as the active indicator */
    }

    /* Style for the ACTIVE tab button in Dark Theme */
    .stApp[data-theme="dark"] button[data-testid="stTab"][aria-selected="true"] {
        color: #8CC63F; /* Bright green for active tab text in dark mode */
        border-bottom: 3px solid #8CC63F; /* Bright green as the active indicator */
    }
</style>
""", unsafe_allow_html=True)

if start_date and end_date:
    # Iterate through each tab and call the display function
    for i, tab_name in enumerate(tab_names):
        with tabs[i]:
            display_dashboard_content(tab_name, start_date, end_date)
else:
    st.info("Select a date range in the sidebar to load and display data.")


# ----------------------------------------
# Footer / Credits
# ----------------------------------------
st.markdown("""
---
Â© 2025 | Crafted with care for clarity, empathy, and forward-looking insights.
""")

# ----------------------------------------
# Performance Tips (shown in sidebar)
# ----------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### ’¡ Performance Tips")
st.sidebar.markdown("""
- Use shorter date ranges for faster loading
- Data is cached for 5 minutes
- Raw view limited to 25,000 records
- Tables show max 1,000 rows for performance
""")

if summary_stats:
    st.sidebar.markdown("### “Š Database Overview")
    st.sidebar.markdown(f"""
    - **Total Records**: {summary_stats.get('total_orders', 0):,}
    - **Date Range**: {summary_stats.get('min_date', 'N/A')} to {summary_stats.get('max_date', 'N/A')}
    - **Avg Order Value**: R{summary_stats.get('avg_order_value', 0):.2f}
    """)