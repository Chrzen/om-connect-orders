# app.py
import streamlit as st
import pandas as pd
import datetime

# Import modularized functions and configurations
from config.config import styling
from utils.data_loader import load_summary_stats, get_date_range
from utils.ui import display_dashboard_content

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Order Data Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply global styling
st.markdown(styling, unsafe_allow_html=True)

# ----------------------------------------
# Initialize Application
# ----------------------------------------
# Load overall summary stats for the sidebar
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

# Get and display the date range filter
min_date, max_date = get_date_range()
if min_date and max_date:
    # Ensure dates are in the correct format
    if isinstance(min_date, (pd.Timestamp, datetime.datetime)): min_date = min_date.date()
    if isinstance(max_date, (pd.Timestamp, datetime.datetime)): max_date = max_date.date()

    # Set default date range to the last 90 days, respecting the available data range
    default_start_attempt = max_date - datetime.timedelta(days=90)
    default_start = max(default_start_attempt, min_date)

    if min_date >= max_date:
        st.sidebar.info(f"Data available only for: {min_date}")
        start_date = end_date = st.sidebar.date_input("Date:", value=min_date, min_value=min_date, max_value=max_date)
    else:
        st.sidebar.info(f"Data available from {min_date} to {max_date}")
        start_date, end_date = st.sidebar.date_input(
            "Select Date Range:", value=[default_start, max_date], min_value=min_date, max_value=max_date
        )
else:
    st.sidebar.error("Unable to load date range from the database.")
    start_date, end_date = None, None

# ----------------------------------------
# Main Content Area
# ----------------------------------------
# Define tab names
tab_names = ["Standard Payments", "SIM Provisioning", "Reward", "Cost Centre", "Airtime", "Raw"]
tabs = st.tabs(tab_names)

# Render content for each tab
if start_date and end_date:
    if start_date > end_date:
        st.error("Error: The start date cannot be after the end date. Please select a valid range.")
    else:
        # Loop through tabs and display content
        for i, tab_name in enumerate(tab_names):
            with tabs[i]:
                display_dashboard_content(tab_name, start_date, end_date)
else:
    st.info("Select a date range in the sidebar to load and display data.")

# ----------------------------------------
# Footer / Credits
# ----------------------------------------
st.divider()
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
- Use shorter date ranges for faster loading.
- Data is cached to improve performance on reruns.
- The "Raw" data view is limited for speed.
""")

if summary_stats:
    st.sidebar.markdown("### Database Overview")
    st.sidebar.markdown(f"""
    - **Total Records**: {summary_stats.get('total_orders', 0):,}
    - **Data Available**: {summary_stats.get('min_date', 'N/A')} to {summary_stats.get('max_date', 'N/A')}
    - **Avg. Order Value**: R{summary_stats.get('avg_order_value', 0):.2f}
    """)