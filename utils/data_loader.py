# utils/data_loader.py
import streamlit as st
import pandas as pd
from config.config import TABLE_NAME

# Import Snowflake connector with proper error handling
try:
    import snowflake.connector as sf
except ImportError as e:
    st.error("Snowflake connector not properly installed. Please run: pip install snowflake-connector-python[pandas]")
    st.stop()

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
    """Load basic summary statistics quickly""" #
    conn = get_fresh_connection()  # Use fresh connection
    if conn is None:
        return {} #

    try:
        # Corrected SQL query using a CROSS JOIN for better compatibility and performance
        sql = f"""
        SELECT * FROM
            (SELECT
                COUNT(*) as total_orders,
                COUNT(DISTINCT SIM_MSISDN) as unique_customers,
                SUM(RETAIL_AMOUNT) as total_revenue,
                AVG(RETAIL_AMOUNT) as avg_order_value
            FROM {TABLE_NAME}
            WHERE PAYMENT_COHORT = 'Sales') AS sales_stats
        CROSS JOIN
            (SELECT
                MIN(CREATED_AT) as min_date,
                MAX(CREATED_AT) as max_date
            FROM {TABLE_NAME}) AS date_stats
        """

        result = pd.read_sql(sql, conn) #
        result.columns = [col.lower() for col in result.columns] #
        return result.iloc[0].to_dict() #

    except Exception as e:
        st.error(f"Error loading summary stats: {e}") #
        # Fallback to a basic query if the main one fails
        try:
            basic_sql = f"SELECT COUNT(*) as total_orders FROM {TABLE_NAME}" #
            basic_result = pd.read_sql(basic_sql, conn) #
            basic_result.columns = [col.lower() for col in basic_result.columns] #
            return {
                'total_orders': basic_result.iloc[0]['total_orders'],
                'unique_customers': 0,
                'total_revenue': 0,
                'avg_order_value': 0,
                'min_date': None,
                'max_date': None
            } #
        except Exception as basic_e:
            st.error(f"Error loading basic stats: {basic_e}") #
            return {} #
    finally:
        if conn:
            conn.close() #

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