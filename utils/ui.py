# utils/ui.py
import streamlit as st
from .data_loader import load_filtered_data, validate_data
from .plotting import (
    plot_daily_kpis, plot_product_type_timeline, plot_weekly_kpis,
    plot_repurchase_propensity, plot_growth_bars, plot_purchase_sequence_sankey,
    plot_bundle_analysis, plot_customer_lifecycle, plot_cohort_analysis,
    plot_geo_demographic_overview, plot_product_correlation_matrix,
    plot_split_sunbursts
)
from config.config import info_texts

def plot_in_a_box(title: str, fig, info_text: str, anchor_id: str):
    """
    A helper function to group a title, an info icon, a chart, and an HTML anchor.
    """
    if fig is None:
        return

    # Add an invisible HTML anchor for the navigation link
    st.markdown(f"<div id='{anchor_id}'></div>", unsafe_allow_html=True)
    
    # The container now gets its styling from the global CSS in config.py
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
        st.plotly_chart(fig, use_container_width=True, key=anchor_id) # Use anchor_id for a unique key

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
        plot_in_a_box(f"Daily Revenue and Orders for {tab}", fig_kpis, "Standard daily KPIs.", f"kpi-{tab}")