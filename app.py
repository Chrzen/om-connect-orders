import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Order Data Dashboard", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ----------------------------------------
# Color Palette
# ----------------------------------------
oldmutual_palette = [
    "#006B54", "#1A8754", "#5AAA46", 
    "#8CC63F", "#00524C", "#E6F2E2"
]
negative_color = "#d65f5f"

# ----------------------------------------
# Data Loading
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_parquet("OrderData_clean.parquet")
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df

df = load_data()

# ----------------------------------------
# Plotly Weekly KPI Function
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

def plot_bundle_size_distribution_plotly(df):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Make a copy so we don’t modify the original
    df = df.copy()

    # 1) Ensure bundleSize_MB exists
    if 'bundleSize_MB' not in df.columns:
        if 'bundleSize' in df.columns:
            df['bundleSize_MB'] = df['bundleSize'] / (1024 * 1024)
        else:
            import json
            def extract_bundle_size(json_str):
                if pd.isna(json_str):
                    return np.nan
                try:
                    details = json.loads(json_str)
                    if 'bundleSize' in details:
                        return details['bundleSize'] / (1024 * 1024)
                    return np.nan
                except:
                    return np.nan

            df['bundleSize_MB'] = df['product_details'].apply(extract_bundle_size)

    # 2) Filter out null or zero bundle sizes
    bundle_data = df.dropna(subset=['bundleSize_MB']).copy()
    bundle_data = bundle_data[bundle_data['bundleSize_MB'] > 0]

    # 3) Define common size bins and labels
    common_sizes = [30, 50, 100, 250, 500, 1024, 2048, 5120, 10240, 20480]
    labels = ['<30MB', '30MB', '50MB', '100MB', '250MB', '500MB',
              '1GB', '2GB', '5GB', '10GB', '20GB']
    bins = [0] + common_sizes + [float('inf')]

    bundle_data['size_category'] = pd.cut(
        bundle_data['bundleSize_MB'],
        bins=bins,
        labels=labels
    )

    # 4) Compute counts by size_category (for histogram)
    size_counts = (
        bundle_data['size_category']
        .value_counts()
        .reindex(labels, fill_value=0)
    )

    # 5) Compute average revenue and counts by size_category (for scatter)
    size_revenue = (
        bundle_data
        .groupby('size_category')['retail_amount']
        .agg(['mean', 'count'])
        .reindex(labels)
        .dropna(subset=['mean'])
    )

    # 6) Popularity over time: top 5 bundles by count
    bundle_data['month'] = bundle_data['created_at'].dt.to_period('M')
    monthly_ct = pd.crosstab(
        bundle_data['month'],
        bundle_data['size_category']
    )
    monthly_ct.index = monthly_ct.index.to_timestamp()
    top_bundles = size_counts.sort_values(ascending=False).head(5).index.tolist()
    monthly_top = monthly_ct[top_bundles].fillna(0)

    # 7) Check if 'age_group' exists for heatmap
    has_age = 'age_group' in bundle_data.columns
    if has_age:
        top_age_groups = (
            bundle_data['age_group']
            .value_counts()
            .head(4)
            .index
            .tolist()
        )
        sub = bundle_data[bundle_data['age_group'].isin(top_age_groups)].copy()
        age_bundle = pd.crosstab(
            sub['age_group'],
            sub['size_category'],
            normalize='index'
        ).reindex(columns=labels, fill_value=0)

    # 8) Build a 2×2 subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Distribution of Data Bundle Sizes",
            "Average Revenue by Bundle Size",
            "Bundle Size Preference by Age Group" if has_age else "Age Group Data Not Available",
            "Bundle Size Popularity Over Time"
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )

    # --- Subplot 1: Histogram (category counts) ---
    # Assign a repeating palette if labels > palette length
    bar_colors = [
        oldmutual_palette[i % len(oldmutual_palette)] 
        for i in range(len(labels))
    ]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[size_counts[label] for label in labels],
            marker_color=bar_colors,
            name="Count"
        ),
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Number of Orders",
        row=1, col=1
    )
    fig.update_xaxes(
        tickangle=45,
        row=1, col=1
    )

    # --- Subplot 2: Scatter + Line (average revenue) ---
    x2 = size_revenue.index.tolist()
    y2 = size_revenue['mean'].tolist()
    marker_sizes = (size_revenue['count'] / 100).tolist()

    fig.add_trace(
        go.Scatter(
            x=x2,
            y=y2,
            mode="markers+lines",
            marker=dict(
                size=marker_sizes,
                color=oldmutual_palette[1],
                sizemode="area",
                opacity=0.7
            ),
            line=dict(color=oldmutual_palette[1], width=2),
            name="Avg Revenue"
        ),
        row=1, col=2
    )
    # Annotate each point with “R{mean:.2f}”
    for idx, label in enumerate(x2):
        fig.add_annotation(
            x=label,
            y=y2[idx],
            text=f"R{y2[idx]:.2f}",
            showarrow=False,
            yshift=10,
            font=dict(size=9),
            xanchor="center"
        )
    fig.update_yaxes(
        title_text="Average Revenue (R)",
        row=1, col=2
    )
    fig.update_xaxes(
        tickangle=45,
        row=1, col=2
    )

    # --- Subplot 3: Heatmap (bundle vs age group) or placeholder text ---
    if has_age:
        fig.add_trace(
            go.Heatmap(
                z=age_bundle.values,
                x=age_bundle.columns.tolist(),
                y=age_bundle.index.tolist(),
                colorscale=[
                    [0, oldmutual_palette[5]],  # lightest
                    [1, oldmutual_palette[0]]   # darkest
                ],
                zmid=age_bundle.values.max() / 2,
                colorbar=dict(
                    title="%",
                    x=0.48,          # shift colorbar so it sits just to the right of subplot
                    y=0.27,          # vertical center for row 2
                    len=0.3,         # height relative to entire figure
                    thickness=15
                )
            ),
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Age Group",
            row=2, col=1
        )
        fig.update_xaxes(
            title_text="Bundle Size",
            row=2, col=1,
            tickangle=45
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[0.5],
                mode="text",
                text=["Age group data not available"],
                textfont=dict(size=12, color="grey"),
                showlegend=False
            ),
            row=2, col=1
        )
        fig.update_xaxes(visible=False, row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)

    # --- Subplot 4: Line plot (popularity over time for top 5) ---
    for idx, bundle in enumerate(top_bundles):
        fig.add_trace(
            go.Scatter(
                x=monthly_top.index,
                y=monthly_top[bundle].values,
                mode="lines+markers",
                marker=dict(size=6, color=oldmutual_palette[idx]),
                line=dict(color=oldmutual_palette[idx], width=2),
                name=str(bundle)
            ),
            row=2, col=2
        )
    fig.update_yaxes(
        title_text="Number of Orders",
        row=2, col=2
    )
    fig.update_xaxes(
        tickangle=45,
        row=2, col=2
    )

    # --- Layout tweaks for all subplots ---
    fig.update_layout(
        height=800,
        showlegend=False,
        margin=dict(t=80, b=50, l=50, r=80)
    )

    return fig

def plot_bundle_size_analysis(df):
    """
    Create comprehensive analysis of data bundle sizes and their popularity, using Plotly.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # 1) Ensure bundleSize_MB exists
    if 'bundleSize_MB' not in df.columns and 'bundleSize' in df.columns:
        df['bundleSize_MB'] = df['bundleSize'] / (1024 * 1024)
    
    # 2) Filter out invalid or zero bundle-size rows
    bundle_data = df.dropna(subset=['bundleSize_MB'])
    bundle_data = bundle_data[bundle_data['bundleSize_MB'] > 0]
    
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
    
    # =============================================================================
    # (A) Top‐left: “Data Bundle Size Distribution” → simple bar chart of counts by size_category
    # =============================================================================
    counts = (
        bundle_data['size_category']
        .value_counts()
        .reindex(all_categories, fill_value=0)
    )
    
    trace_dist = go.Bar(
        x=all_categories,
        y=counts.values,
        marker_color=oldmutual_palette[0],
        text=[f"{int(v):,}" for v in counts.values],
        textposition='outside',
        name='Count'
    )
    
    # =============================================================================
    # (B) Top‐right: “Bundle Size Preference by Age Group” → heatmap of [% by age_group × size_category]
    # =============================================================================
    if 'age_group' in bundle_data.columns:
        top_age_groups = (
            bundle_data['age_group']
            .value_counts()
            .nlargest(5)
            .index
            .tolist()
        )
        age_bundle = bundle_data[bundle_data['age_group'].isin(top_age_groups)]
        
        # Cross‐tabulate (rows=age_group, cols=size_category), normalized by row
        age_bundle_cross = (
            pd.crosstab(
                age_bundle['age_group'],
                age_bundle['size_category'],
                normalize='index'
            )
            .reindex(index=top_age_groups, columns=all_categories, fill_value=0)
            * 100  # convert to percentage
        )
        
        # z‐values as 2D list of percentages
        z_values = age_bundle_cross.values.tolist()
        x_labels = all_categories
        y_labels = top_age_groups
        
        trace_heat = go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale=oldmutual_palette,  # reuse your palette as a discrete colorscale
            hovertemplate='%{y}<br>%{x}: %{z:.1f}%<extra></extra>',
            showscale=True,
            colorbar=dict(title='Percentage')
        )
    else:
        trace_heat = None
    
    # =============================================================================
    # (C) Bottom‐left: “Revenue Metrics by Bundle Size” → dual‐axis: (1) avg_revenue as bars, (2) total_revenue as line
    # =============================================================================
    revenue_by_size = (
        bundle_data.groupby('size_category')
        .agg(avg_revenue=('retail_amount', 'mean'),
             total_revenue=('retail_amount', 'sum'))
        .reindex(all_categories, fill_value=0)
        .reset_index()
    )
    
    trace_avg_rev = go.Bar(
        x=revenue_by_size['size_category'],
        y=revenue_by_size['avg_revenue'],
        marker_color=oldmutual_palette[1],
        name='Avg. Revenue (R)',
        text=[f"R{v:.2f}" for v in revenue_by_size['avg_revenue']],
        textposition='outside'
    )
    
    trace_total_rev = go.Scatter(
        x=revenue_by_size['size_category'],
        y=revenue_by_size['total_revenue'],
        mode='lines+markers',
        marker=dict(color=oldmutual_palette[2]),
        line=dict(width=2),
        name='Total Revenue (R)',
        yaxis='y2'
    )
    
    # =============================================================================
    # (D) Bottom‐right: “Bundle Size Popularity Over Time” → line plots for top 5 popular bundles
    # =============================================================================
    top_bundles = (
        bundle_data['size_category']
        .value_counts()
        .nlargest(5)
        .index
        .tolist()
    )
    
    # Convert created_at to Period('M') and then back to timestamp
    bundle_data['month'] = bundle_data['created_at'].dt.to_period('M').dt.to_timestamp()
    
    monthly_ct = (
        pd.crosstab(
            bundle_data['month'],
            bundle_data['size_category']
        )
        .reindex(columns=all_categories, fill_value=0)
    )
    
    # Keep only the top 5 bundles
    monthly_ct = monthly_ct[top_bundles].copy()
    
    traces_time = []
    for idx, b in enumerate(top_bundles):
        traces_time.append(
            go.Scatter(
                x=monthly_ct.index,
                y=monthly_ct[b],
                mode='lines+markers',
                line=dict(color=oldmutual_palette[idx % len(oldmutual_palette)], width=2),
                name=str(b)
            )
        )
    
    # =============================================================================
    # Build a 2×2 subplot grid
    # =============================================================================
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "bar"}, {"type": "heatmap"}],
            [{"secondary_y": True}, {"type": "scatter"}]
        ],
        subplot_titles=(
            "Data Bundle Size Distribution",
            "Bundle Size Preference by Age Group",
            "Revenue Metrics by Bundle Size",
            "Bundle Size Popularity Over Time"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # (A) row=1, col=1  → distribution bar
    fig.add_trace(trace_dist, row=1, col=1)
    fig.update_xaxes(
        tickangle=45,
        row=1,
        col=1
    )
    fig.update_yaxes(
        title_text="Number of Purchases",
        row=1,
        col=1
    )
    
    # (B) row=1, col=2  → heatmap (if available)
    if trace_heat:
        fig.add_trace(trace_heat, row=1, col=2)
        fig.update_xaxes(
            tickangle=45,
            row=1,
            col=2
        )
        fig.update_yaxes(
            title_text="Age Group",
            row=1,
            col=2
        )
    else:
        # Insert an empty annotation if age_group is missing
        fig.add_annotation(
            text="Age group data not available",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            showarrow=False,
            row=1, col=2
        )
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
    
    # (C) row=2, col=1  → avg_revenue bar on primary y; total_revenue line on secondary y
    fig.add_trace(trace_avg_rev, row=2, col=1, secondary_y=False)
    fig.add_trace(trace_total_rev, row=2, col=1, secondary_y=True)
    
    fig.update_xaxes(
        tickangle=45,
        row=2,
        col=1
    )
    fig.update_yaxes(
        title_text="Avg. Revenue (R)",
        row=2,
        col=1,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Total Revenue (R)",
        row=2,
        col=1,
        secondary_y=True,
        title_font=dict(color=oldmutual_palette[2]),
        tickfont=dict(color=oldmutual_palette[2])
    )
    
    # (D) row=2, col=2  → time‐series lines for each top bundle
    for tr in traces_time:
        fig.add_trace(tr, row=2, col=2)
    fig.update_xaxes(
        tickangle=45,
        row=2,
        col=2,
        tickformat="%b %Y"
    )
    fig.update_yaxes(
        title_text="Number of Purchases",
        row=2,
        col=2
    )
    
    # =============================================================================
    # Final layout tweaks
    # =============================================================================
    fig.update_layout(
        showlegend=True,
        height=800,
        width=1400,
        margin=dict(t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig


def plot_customer_lifecycle_value(df):
    """
    Create visualizations showing customer behavior and value across their lifecycle, using Plotly.
    """
    df = df.copy()

    # 1) Calculate days since acquisition
    df['acquisition_date'] = df.groupby('customer_id_number')['customer_created_at'].transform('min')
    df['days_since_acquisition'] = (df['created_at'] - df['acquisition_date']).dt.days

    # 2) Create lifecycle stage categories
    df['lifecycle_stage'] = pd.cut(
        df['days_since_acquisition'],
        bins=[-1, 7, 30, 90, 180, 365, float('inf')],
        labels=['First Week', '8-30 Days', '31-90 Days', '91-180 Days', '181-365 Days', '365+ Days']
    )

    # 3) Compute metrics by stage
    stage_metrics = (
        df.groupby('lifecycle_stage')
          .agg(
              avg_order_value=('retail_amount', 'mean'),
              num_orders=('id', 'count'),
              num_customers=('customer_id_number', pd.Series.nunique)
          )
          .reset_index()
    )
    stage_metrics['orders_per_customer'] = (
        stage_metrics['num_orders'] / stage_metrics['num_customers']
    )

    # 4) Weekly & cumulative revenue
    df['week_since_acquisition'] = (df['days_since_acquisition'] // 7) + 1
    weekly_revenue = df.groupby('week_since_acquisition')['retail_amount'].sum().sort_index()
    cumulative_revenue = weekly_revenue.cumsum()

    # 5) Product category preference
    df['product_category'] = df['product_name'].str.split().str[0]
    top_categories = df['product_category'].value_counts().head(5).index.tolist()
    category_data = df[df['product_category'].isin(top_categories)]
    category_stage_cross = (
        pd.crosstab(
            category_data['lifecycle_stage'],
            category_data['product_category'],
            normalize='index'
        )
        * 100
    ).reindex(
        index=stage_metrics['lifecycle_stage'].astype(str),
        columns=top_categories,
        fill_value=0
    )

    # 6) Average bundle size by stage (if available)
    has_bundle = 'bundleSize_MB' in df.columns
    if has_bundle:
        bundle_data = df.dropna(subset=['bundleSize_MB'])
        bundle_data = bundle_data[bundle_data['bundleSize_MB'] > 0]
        bundle_by_stage = (
            bundle_data.groupby('lifecycle_stage')['bundleSize_MB']
                       .mean()
                       .reset_index()
        )
        # Convert to display units
        bundle_by_stage['display_size'] = bundle_by_stage['bundleSize_MB']
        bundle_by_stage['unit'] = 'MB'
        gb_mask = bundle_by_stage['bundleSize_MB'] >= 1024
        bundle_by_stage.loc[gb_mask, 'display_size'] = (
            bundle_by_stage.loc[gb_mask, 'bundleSize_MB'] / 1024
        )
        bundle_by_stage.loc[gb_mask, 'unit'] = 'GB'
    else:
        daily_avg = (
            df[df['days_since_acquisition'] <= 30]
            .groupby('days_since_acquisition')['retail_amount']
            .mean()
            .reset_index()
        )

    # =============================================================================
    # Build a 3×2 subplot grid
    # =============================================================================
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"colspan": 2, "type": "scatter"}, None],
            [{"type": "heatmap"}, {"type": "bar" if has_bundle else "scatter"}]
        ],
        subplot_titles=(
            "Average Order Value by Lifecycle Stage",
            "Orders per Customer by Lifecycle Stage",
            "Cumulative Revenue by Customer Tenure",
            "",
            "Product Category Preference by Lifecycle Stage",
            "Average Bundle Size by Lifecycle Stage" if has_bundle else "Avg. Purchase Amount (First Month)"
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
        row_heights=[0.25, 0.35, 0.25]
    )

    # -------------------------------------------------------------------------
    # (A) Row 1, Col 1 → Average Order Value by Lifecycle Stage
    # -------------------------------------------------------------------------
    bar_avg = go.Bar(
        x=stage_metrics['lifecycle_stage'].astype(str),
        y=stage_metrics['avg_order_value'],
        marker_color=oldmutual_palette[0],
        name='Avg Order Value',
        text=[f"R{v:.2f}" for v in stage_metrics['avg_order_value']],
        textposition='outside'
    )
    fig.add_trace(bar_avg, row=1, col=1)
    fig.update_xaxes(
        tickangle=45,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="R",
        row=1, col=1
    )

    # -------------------------------------------------------------------------
    # (B) Row 1, Col 2 → Orders per Customer by Lifecycle Stage
    # -------------------------------------------------------------------------
    bar_orders = go.Bar(
        x=stage_metrics['lifecycle_stage'].astype(str),
        y=stage_metrics['orders_per_customer'],
        marker_color=oldmutual_palette[1],
        name='Orders per Customer',
        text=[f"{v:.2f}" for v in stage_metrics['orders_per_customer']],
        textposition='outside'
    )
    fig.add_trace(bar_orders, row=1, col=2)
    fig.update_xaxes(
        tickangle=45,
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Orders/Customer",
        row=1, col=2
    )

    # -------------------------------------------------------------------------
    # (C) Row 2, Col 1–2 (colspan) → Cumulative Revenue by Customer Tenure
    # -------------------------------------------------------------------------
    line_cum = go.Scatter(
        x=cumulative_revenue.index,
        y=cumulative_revenue.values,
        mode='lines+markers',
        line=dict(color=oldmutual_palette[2], width=2),
        fill='tonexty',
        name='Cumulative Revenue'
    )
    fig.add_trace(line_cum, row=2, col=1)

    # Annotations: Week 1 & Week 4
    if 1 in cumulative_revenue.index:
        fig.add_annotation(
            x=1,
            y=cumulative_revenue.loc[1],
            text=f"Week 1: R{cumulative_revenue.loc[1]:,.0f}",
            showarrow=True,
            arrowhead=2,
            ax=0, ay=-40,
            row=2, col=1
        )
    if 4 in cumulative_revenue.index:
        fig.add_annotation(
            x=4,
            y=cumulative_revenue.loc[4],
            text=f"Month 1: R{cumulative_revenue.loc[4]:,.0f}",
            showarrow=True,
            arrowhead=2,
            ax=0, ay=40,
            row=2, col=1
        )

    # Vertical month markers at weeks [4,8,12,16,20,24]
    max_rev = cumulative_revenue.max()
    shapes = []
    month_texts = []
    for wk in [4, 8, 12, 16, 20, 24]:
        if wk in cumulative_revenue.index:
            shapes.append({
                "type": "line",
                "x0": wk,
                "y0": 0,
                "x1": wk,
                "y1": max_rev,
                "xref": "x3",    # row=2, col=1 → xaxis3
                "yref": "y3",    # row=2, col=1 → yaxis3
                "line": {"dash": "dash", "color": "gray", "width": 1},
                "opacity": 0.3
            })
            month_texts.append({
                "x": wk,
                "y": max_rev * 0.05,
                "xref": "x3",
                "yref": "y3",
                "text": f"Month {wk//4}",
                "showarrow": False,
                "font": {"size": 10, "color": "black"},
                "bgcolor": "white",
                "opacity": 0.7
            })
    fig.update_layout(shapes=shapes)
    for ann in month_texts:
        fig.add_annotation(**ann)

    fig.update_xaxes(
        title_text="Weeks Since Acquisition",
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Cumulative Revenue (R)",
        row=2, col=1,
        tickprefix="R",
        tickformat=","
    )

    # -------------------------------------------------------------------------
    # (D) Row 3, Col 1 → Product Category Preference by Lifecycle Stage (Heatmap)
    # -------------------------------------------------------------------------
    heat = go.Heatmap(
        z=category_stage_cross.values,
        x=category_stage_cross.columns.tolist(),
        y=category_stage_cross.index.tolist(),
        colorscale=oldmutual_palette,
        hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        showscale=True,
        colorbar=dict(title="%")
    )
    fig.add_trace(heat, row=3, col=1)
    fig.update_xaxes(
        title_text="Product Category",
        tickangle=45,
        row=3, col=1
    )
    fig.update_yaxes(
        title_text="Lifecycle Stage",
        row=3, col=1
    )

    # -------------------------------------------------------------------------
    # (E) Row 3, Col 2 → Average Bundle Size by Lifecycle Stage (or fallback)
    # -------------------------------------------------------------------------
    if has_bundle:
        bar_bundle = go.Bar(
            x=bundle_by_stage['lifecycle_stage'].astype(str),
            y=bundle_by_stage['display_size'],
            marker_color=oldmutual_palette[3],
            name='Avg Bundle Size',
            text=[f"{row.display_size:.1f} {row.unit}" for _, row in bundle_by_stage.iterrows()],
            textposition='outside'
        )
        fig.add_trace(bar_bundle, row=3, col=2)
        fig.update_xaxes(
            tickangle=45,
            row=3, col=2
        )
        fig.update_yaxes(
            title_text="Average Bundle Size",
            row=3, col=2
        )
    else:
        line_daily = go.Scatter(
            x=daily_avg['days_since_acquisition'],
            y=daily_avg['retail_amount'],
            mode='lines+markers',
            line=dict(color=oldmutual_palette[3], width=2),
            name='Avg Purchase Amount'
        )
        fig.add_trace(line_daily, row=3, col=2)
        fig.update_xaxes(
            title_text="Days Since Acquisition",
            row=3, col=2
        )
        fig.update_yaxes(
            title_text="Avg Purchase Amount (R)",
            row=3, col=2,
            tickprefix="R",
            tickformat=","
        )

    # =============================================================================
    # Final layout adjustments
    # =============================================================================
    fig.update_layout(
        showlegend=False,
        height=900,
        width=1400,
        margin=dict(t=100, b=60, l=60, r=60),
        font=dict(family="Arial, sans-serif", size=12)
    )

    return fig

def plot_product_relationships(df):
    """
    Create Plotly visualizations showing product relationships and purchasing patterns.
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Make a copy to avoid side effects
    df = df.copy()

    # =============================================================================
    # 1) Product Success Matrix Data (Top-left)
    # =============================================================================
    product_metrics = (
        df.groupby('product_name')
          .agg(
              order_count=('id', 'count'),
              avg_price=('retail_amount', 'mean'),
              total_revenue=('retail_amount', 'sum')
          )
          .reset_index()
    )
    top_products = product_metrics.nlargest(15, 'order_count').reset_index(drop=True)
    # Marker sizes scaled by total_revenue
    max_rev = top_products['total_revenue'].max()
    sizes = (top_products['total_revenue'] / max_rev) * 50 + 10  # scale to [10, 60]

    # =============================================================================
    # 2) Product Purchase Sequence Data → use a Sankey to show first→second transitions (Top-right)
    # =============================================================================
    multi_df = df.groupby('customer_id_number').filter(lambda x: len(x) > 1).copy()
    sankey_available = len(multi_df) > 0

    if sankey_available:
        multi_df = multi_df.sort_values(['customer_id_number', 'created_at'])
        firsts = multi_df.groupby('customer_id_number').first().reset_index()
        top_first_counts = firsts['product_name'].value_counts().head(5)
        top_firsts = top_first_counts.index.tolist()

        second_transitions = []
        for first_prod in top_firsts:
            custs = firsts[firsts['product_name'] == first_prod]['customer_id_number']
            seconds = []
            for c in custs:
                purchases = multi_df[multi_df['customer_id_number'] == c]
                if len(purchases) > 1:
                    seconds.append(purchases.iloc[1]['product_name'])
            if seconds:
                counts = pd.Series(seconds).value_counts().head(3)
                for sec_prod, cnt in counts.items():
                    second_transitions.append({
                        'first': first_prod,
                        'second': sec_prod,
                        'count': cnt
                    })

        if second_transitions:
            trans_df = pd.DataFrame(second_transitions)
            all_nodes = list(dict.fromkeys(
                trans_df['first'].tolist() + trans_df['second'].tolist()
            ))
            node_indices = {n: i for i, n in enumerate(all_nodes)}

            sankey_source = trans_df['first'].map(node_indices).tolist()
            sankey_target = trans_df['second'].map(node_indices).tolist()
            sankey_value  = trans_df['count'].tolist()

            sankey_node = dict(
                label=all_nodes,
                pad=15,
                thickness=20,
                color=[oldmutual_palette[i % len(oldmutual_palette)] for i in range(len(all_nodes))]
            )
            sankey_link = dict(
                source=sankey_source,
                target=sankey_target,
                value=sankey_value,
                color=oldmutual_palette[2]
            )
        else:
            sankey_available = False

    # =============================================================================
    # 3) Product Association (Co‐occurrence) Data (Middle-left)
    # =============================================================================
    cust_prod_sets = df.groupby('customer_id_number')['product_name'].apply(set)
    cust_prod_sets = cust_prod_sets[cust_prod_sets.apply(lambda s: len(s) > 1)]
    cooccurrence_available = len(cust_prod_sets) > 0

    if cooccurrence_available:
        all_prods = set()
        for s in cust_prod_sets:
            all_prods.update(s)
        prod_counts = df['product_name'].value_counts().head(10)
        top10 = prod_counts.index.tolist()

        # Build co-occurrence matrix
        co_mat = np.zeros((10, 10), dtype=float)
        for s in cust_prod_sets:
            for i, p1 in enumerate(top10):
                if p1 in s:
                    for j, p2 in enumerate(top10):
                        if p2 in s:
                            co_mat[i, j] += 1
        co_df = pd.DataFrame(co_mat, index=top10, columns=top10).fillna(0)

        # mask upper triangle
        z_vals = co_df.values.copy()
        mask = np.triu_indices_from(z_vals)
        z_vals[mask] = np.nan
    else:
        z_vals = None

    # =============================================================================
    # 4) Purchase Timing Data (Middle-right)
    # =============================================================================
    df_sorted = df.sort_values(['customer_id_number', 'created_at'])
    df_sorted['next_purchase'] = df_sorted.groupby('customer_id_number')['created_at'].shift(-1)
    df_sorted['days_to_next'] = (df_sorted['next_purchase'] - df_sorted['created_at']).dt.days
    valid_intervals = df_sorted.dropna(subset=['days_to_next'])
    valid_intervals = valid_intervals[(valid_intervals['days_to_next'] > 0) & (df_sorted['days_to_next'] <= 90)]
    hist_available = len(valid_intervals) > 0
    if hist_available:
        median_days = valid_intervals['days_to_next'].median()

    # =============================================================================
    # 5) Product Popularity Over Time (Bottom, full width)
    # =============================================================================
    top5_overall = df['product_name'].value_counts().head(5).index.tolist()
    df['week'] = df['created_at'].dt.to_period('W').dt.to_timestamp()
    weekly_ct = pd.crosstab(df['week'], df['product_name']).reindex(columns=top5_overall, fill_value=0)

    # =============================================================================
    # Build Plotly Subplots: 3 rows × 2 cols, with last row spanning both cols
    # =============================================================================
    specs = [
        [{"type": "scatter"}, {"type": "domain"}],
        [{"type": "heatmap"}, {"type": "xy"}],
        [{"colspan": 2, "type": "scatter"}, None]
    ]
    subplot_titles = [
        "Product Success Matrix: Volume vs. Price",
        "Product Purchase Sequence (Sankey)",
        "Product Co-occurrence Matrix",
        "Days Between Purchases",
        "Weekly Product Popularity Trends"
    ]
    fig = make_subplots(
        rows=3, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
        row_heights=[0.30, 0.30, 0.30]
    )

    # -------------------------------------------------------------------------
    # (1) Row1,Col1: Bubble Scatter (order_count vs avg_price, size ~ total_revenue)
    # -------------------------------------------------------------------------
    scatter1 = go.Scatter(
        x=top_products['order_count'],
        y=top_products['avg_price'],
        mode='markers',
        marker=dict(
            size=sizes.tolist(),
            color=[oldmutual_palette[i % len(oldmutual_palette)] for i in range(len(top_products))],
            opacity=0.7,
            line=dict(width=1, color='DarkGray')
        ),
        text=top_products['product_name'],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Orders: %{x}<br>"
            "Avg Price: R%{y:.2f}<br>"
            "Total Rev: R%{marker.size:,}<extra></extra>"
        )
    )
    fig.add_trace(scatter1, row=1, col=1)
    fig.update_xaxes(title_text="Number of Orders", row=1, col=1)
    fig.update_yaxes(title_text="Average Price (R)", row=1, col=1)

    # -------------------------------------------------------------------------
    # (2) Row1,Col2: Sankey (first→second purchase)
    # -------------------------------------------------------------------------
    if sankey_available and second_transitions:
        sankey = go.Sankey(
            node=sankey_node,
            link=sankey_link,
            domain=dict(x=[0, 1], y=[0, 1])
        )
        fig.add_trace(sankey, row=1, col=2)
    else:
        fig.add_annotation(
            text="Insufficient data for purchase sequence",
            x=0.5, y=0.5,
            xref="x domain", yref="y domain",
            showarrow=False,
            row=1, col=2
        )

    # -------------------------------------------------------------------------
    # (3) Row2,Col1: Heatmap (co-occurrence)
    # -------------------------------------------------------------------------
    if cooccurrence_available:
        heatmap = go.Heatmap(
            z=z_vals,
            x=co_df.columns.tolist(),
            y=co_df.index.tolist(),
            colorscale=oldmutual_palette,
            hovertemplate="%{y} & %{x}: %{z:.0f}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Co-occurrence")
        )
        fig.add_trace(heatmap, row=2, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_yaxes(title_text="Product", row=2, col=1)
    else:
        fig.add_annotation(
            text="Insufficient data for co-occurrence",
            x=0.5, y=0.5,
            xref="x domain", yref="y domain",
            showarrow=False,
            row=2, col=1
        )

    # -------------------------------------------------------------------------
    # (4) Row2,Col2: Histogram (days between purchases)
    # -------------------------------------------------------------------------
    if hist_available:
        hist = go.Histogram(
            x=valid_intervals['days_to_next'],
            nbinsx=30,
            marker_color=oldmutual_palette[3],
            opacity=0.7,
            name="Interval"
        )
        fig.add_trace(hist, row=2, col=2)

        # Vertical line for median
        fig.add_shape(
            type="line",
            x0=median_days, x1=median_days,
            y0=0, y1=valid_intervals['days_to_next'].value_counts().max(),
            xref="x2", yref="y2",
            line=dict(color="red", dash="dash", width=2)
        )
        fig.add_annotation(
            x=median_days,
            y=valid_intervals['days_to_next'].value_counts().max() * 0.9,
            text=f"Median: {median_days:.1f} days",
            showarrow=False,
            bgcolor="white",
            bordercolor="gray",
            font=dict(size=10),
            row=2, col=2
        )
        fig.update_xaxes(title_text="Days Between Purchases", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
    else:
        fig.add_annotation(
            text="Insufficient data for interpurchase time",
            x=0.5, y=0.5,
            xref="x domain", yref="y domain",
            showarrow=False,
            row=2, col=2
        )

    # -------------------------------------------------------------------------
    # (5) Row3,Col1–2: Line charts (weekly product popularity)
    # -------------------------------------------------------------------------
    for i, product in enumerate(top5_overall):
        fig.add_trace(
            go.Scatter(
                x=weekly_ct.index,
                y=weekly_ct[product],
                mode='lines+markers',
                line=dict(color=oldmutual_palette[i % len(oldmutual_palette)], width=2),
                name=product,
                showlegend=(i == 0)  # only show legend once
            ),
            row=3, col=1
        )
    fig.update_xaxes(
        tickangle=45,
        tickformat="%b %-d",
        row=3, col=1
    )
    fig.update_yaxes(title_text="Number of Orders", row=3, col=1)

    # Move the trace entries to a shared legend at bottom
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )

    # =============================================================================
    # Final layout tweaks
    # =============================================================================
    fig.update_layout(
        height=1000,
        width=1400,
        margin=dict(t=80, b=60, l=60, r=60),
        font=dict(family="Arial, sans-serif", size=12)
    )

    return fig

def plot_temporal_patterns(df):
    """
    Create Plotly visualizations of temporal patterns and seasonality in the data.
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Make a copy to avoid side effects
    df = df.copy()

    # -----------------------------------------------------------------------------
    # Weekly metrics (resample by week ending Sunday)
    # -----------------------------------------------------------------------------
    df['week'] = df['created_at'].dt.to_period('W-SUN').dt.to_timestamp()
    weekly_orders = df.groupby('week').size().rename('orders')
    weekly_revenue = df.groupby('week')['retail_amount'].sum().rename('revenue')
    weekly_customers = df.groupby('week')['customer_id_number'].nunique().rename('customers')
    weekly_avg_order = df.groupby('week')['retail_amount'].mean().rename('avg_order')

    # -----------------------------------------------------------------------------
    # Day-of-week metrics
    # -----------------------------------------------------------------------------
    df['day_of_week'] = df['created_at'].dt.day_name()
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_metrics = (
        df.groupby('day_of_week')
          .agg(
              orders=('id','count'),
              revenue=('retail_amount','sum'),
              customers=('customer_id_number', pd.Series.nunique)
          )
          .reindex(day_order)
          .reset_index()
    )

    # -----------------------------------------------------------------------------
    # Hour-of-day metrics
    # -----------------------------------------------------------------------------
    df['hour'] = df['created_at'].dt.hour
    hour_metrics = (
        df.groupby('hour')
          .agg(
              orders=('id','count'),
              revenue=('retail_amount','sum'),
              avg_value=('retail_amount','mean')
          )
          .reset_index()
    )

    # -----------------------------------------------------------------------------
    # Day-of-month metrics
    # -----------------------------------------------------------------------------
    df['day_of_month'] = df['created_at'].dt.day
    dom_metrics = (
        df.groupby('day_of_month')
          .agg(
              orders=('id','count'),
              revenue=('retail_amount','sum')
          )
          .reset_index()
    )
    # Linear trend for day-of-month orders
    z = np.polyfit(dom_metrics['day_of_month'], dom_metrics['orders'], 1)
    p = np.poly1d(z)
    dom_metrics['trend'] = p(dom_metrics['day_of_month'])

    # -----------------------------------------------------------------------------
    # Month-over-month growth
    # -----------------------------------------------------------------------------
    df['month'] = df['created_at'].dt.to_period('M').dt.to_timestamp()
    monthly_orders = df.groupby('month').size().rename('orders')
    monthly_revenue = df.groupby('month')['retail_amount'].sum().rename('revenue')
    mo_monthly = pd.concat([monthly_orders, monthly_revenue], axis=1).sort_index()
    mo_monthly['orders_pct'] = mo_monthly['orders'].pct_change() * 100
    mo_monthly['revenue_pct'] = mo_monthly['revenue'].pct_change() * 100

    # =============================================================================
    # Build Plotly subplots: 3 rows × 2 cols, specifying which subplots support secondary_y
    # =============================================================================
    specs = [
        [{"type": "xy", "secondary_y": True}, {"type": "xy", "secondary_y": True}],
        [{"type": "xy", "secondary_y": True}, {"type": "xy", "secondary_y": True}],
        [{"type": "xy", "secondary_y": False}, {"type": "xy", "secondary_y": True}]
    ]
    subplot_titles = [
        "Weekly Orders & Revenue",
        "Weekly Customers & Avg Order Value",
        "Orders & Revenue by Day of Week",
        "Orders & Avg Value by Hour of Day",
        "Order Volume by Day of Month",
        "Month-over-Month Growth"
    ]

    fig = make_subplots(
        rows=3, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.10
    )

    # ////////////////////////////////////////////////////////////////////////////
    # Row 1, Col 1 → Weekly Orders (bar) & Revenue (line on secondary y)
    # ////////////////////////////////////////////////////////////////////////////
    fig.add_trace(
        go.Bar(
            x=weekly_orders.index,
            y=weekly_orders.values,
            marker_color=oldmutual_palette[0],
            name="Orders (weekly)",
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
            name="Revenue (R)",
            hovertemplate="Revenue: R%{y:,.0f}<extra></extra>"
        ),
        row=1, col=1, secondary_y=True
    )
    fig.update_xaxes(
        tickformat="%b %d",
        tickangle=45,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Orders",
        row=1, col=1,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Revenue (R)",
        row=1, col=1,
        secondary_y=True,
        tickprefix="R"
    )

    # ////////////////////////////////////////////////////////////////////////////
    # Row 1, Col 2 → Weekly Customers (bar) & Avg Order (line on secondary y)
    # ////////////////////////////////////////////////////////////////////////////
    fig.add_trace(
        go.Bar(
            x=weekly_customers.index,
            y=weekly_customers.values,
            marker_color=oldmutual_palette[2],
            name="Unique Customers",
            opacity=0.7
        ),
        row=1, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=weekly_avg_order.index,
            y=weekly_avg_order.values,
            mode='lines+markers',
            line=dict(color=oldmutual_palette[3], width=2),
            name="Avg Order Value (R)",
            hovertemplate="Avg: R%{y:.2f}<extra></extra>"
        ),
        row=1, col=2, secondary_y=True
    )
    fig.update_xaxes(
        tickformat="%b %d",
        tickangle=45,
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Customers",
        row=1, col=2,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Avg Order Value (R)",
        row=1, col=2,
        secondary_y=True,
        tickprefix="R"
    )

    # ////////////////////////////////////////////////////////////////////////////
    # Row 2, Col 1 → Orders (bar) & Revenue (line) by Day of Week
    # ////////////////////////////////////////////////////////////////////////////
    fig.add_trace(
        go.Bar(
            x=day_metrics['day_of_week'],
            y=day_metrics['orders'],
            marker_color=oldmutual_palette[0],
            name="Orders",
            opacity=0.8
        ),
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=day_metrics['day_of_week'],
            y=day_metrics['revenue'],
            mode='lines+markers',
            line=dict(color=oldmutual_palette[1], width=2),
            name="Revenue (R)",
            hovertemplate="Revenue: R%{y:,.0f}<extra></extra>"
        ),
        row=2, col=1, secondary_y=True
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=day_order,
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Orders",
        row=2, col=1,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Revenue (R)",
        row=2, col=1,
        secondary_y=True,
        tickprefix="R"
    )

    # ////////////////////////////////////////////////////////////////////////////
    # Row 2, Col 2 → Orders (bar) & Avg Value (line) by Hour of Day
    # ////////////////////////////////////////////////////////////////////////////
    fig.add_trace(
        go.Bar(
            x=hour_metrics['hour'],
            y=hour_metrics['orders'],
            marker_color=oldmutual_palette[2],
            name="Orders",
            opacity=0.8
        ),
        row=2, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=hour_metrics['hour'],
            y=hour_metrics['avg_value'],
            mode='lines+markers',
            line=dict(color=oldmutual_palette[3], width=2),
            name="Avg Order Value (R)",
            hovertemplate="Avg: R%{y:.2f}<extra></extra>"
        ),
        row=2, col=2, secondary_y=True
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(0,24,2)),
        row=2, col=2
    )
    fig.update_yaxes(
        title_text="Orders",
        row=2, col=2,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Avg Order Value (R)",
        row=2, col=2,
        secondary_y=True,
        tickprefix="R"
    )

    # ////////////////////////////////////////////////////////////////////////////
    # Row 3, Col 1 → Day-of-Month Orders (bar) with trendline
    # ////////////////////////////////////////////////////////////////////////////
    fig.add_trace(
        go.Bar(
            x=dom_metrics['day_of_month'],
            y=dom_metrics['orders'],
            marker_color=oldmutual_palette[4],
            name="Orders",
            opacity=0.8
        ),
        row=3, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=dom_metrics['day_of_month'],
            y=dom_metrics['trend'],
            mode='lines',
            line=dict(color=oldmutual_palette[1], dash='dash', width=2),
            name="Trend",
            hovertemplate="Trend: %{y:.2f}<extra></extra>"
        ),
        row=3, col=1, secondary_y=False
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1,32,2)),
        row=3, col=1
    )
    fig.update_yaxes(
        title_text="Orders",
        row=3, col=1,
        secondary_y=False
    )

    # ////////////////////////////////////////////////////////////////////////////
    # Row 3, Col 2 → Month-over-Month Growth: Orders (bar) & Revenue (line)
    # ////////////////////////////////////////////////////////////////////////////
    fig.add_trace(
        go.Bar(
            x=mo_monthly.index,
            y=mo_monthly['orders_pct'],
            marker_color=oldmutual_palette[0],
            name="% Orders Growth",
            opacity=0.7
        ),
        row=3, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=mo_monthly.index,
            y=mo_monthly['revenue_pct'],
            mode='lines+markers',
            line=dict(color=oldmutual_palette[1], width=2),
            name="% Revenue Growth",
            hovertemplate="%{y:.1f}%<extra></extra>"
        ),
        row=3, col=2, secondary_y=True
    )
    fig.add_shape(
        type="line",
        x0=mo_monthly.index.min(),
        x1=mo_monthly.index.max(),
        y0=0, y1=0,
        xref="x6", yref="y6",
        line=dict(color="gray", dash="solid", width=1)
    )
    fig.update_xaxes(
        tickformat="%b %Y",
        tickangle=45,
        row=3, col=2
    )
    fig.update_yaxes(
        title_text="% Orders Growth",
        row=3, col=2,
        secondary_y=False,
        ticksuffix="%"
    )
    fig.update_yaxes(
        title_text="% Revenue Growth",
        row=3, col=2,
        secondary_y=True,
        ticksuffix="%"
    )

    # =============================================================================
    # Final layout tweaks
    # =============================================================================
    fig.update_layout(
        height=1000,
        width=1400,
        margin=dict(t=100, b=60, l=60, r=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        font=dict(family="Arial, sans-serif", size=12)
    )

    return fig

def add_province_from_postal_code(df):
    """
    Derive 'province' from South African postal code (customer_address5).
    """
    import pandas as pd

    


def plot_geographic_analysis(df):
    """
    Create Plotly visualizations of geographic patterns using derived 'province' data.
    """
    if 'customer_address5' not in df.columns:
        print("Warning: 'customer_address5' column not found in dataframe")
        return df.copy()

    df_copy = df.copy()
    postal_ranges = {
        'Western Cape': [(7000, 8099)],
        'Eastern Cape': [(5200, 6499)],
        'Northern Cape': [(8100, 8999)],
        'Free State': [(9300, 9999)],
        'North West': [(2500, 2899)],
        'Gauteng': [(1, 2499)],
        'Mpumalanga': [(1200, 1399)],
        'Limpopo': [(700, 999)],
        'KwaZulu-Natal': [(3200, 4999)]
    }

    def get_province(postal_code):
        if pd.isna(postal_code):
            return 'Unknown'
        try:
            postal_int = int(postal_code)
            for prov, ranges in postal_ranges.items():
                for start, end in ranges:
                    if start <= postal_int <= end:
                        return prov
            return 'Unknown'
        except (ValueError, TypeError):
            return 'Unknown'

    df_copy['derived_province'] = df_copy['customer_address5'].apply(get_province)
    if 'province' in df_copy.columns:
        mask = df_copy['province'].isna()
        df_copy.loc[mask, 'province'] = df_copy.loc[mask, 'derived_province']
    else:
        df_copy['province'] = df_copy['derived_province']
    df_copy.drop(columns=['derived_province'], inplace=True)
    
    df = df_copy

    # Ensure 'province' is populated
    if 'province' not in df.columns or df['province'].isna().sum() > 0.8 * len(df):
        df = add_province_from_postal_code(df)

    # Compute metrics by province
    province_revenue = df.groupby('province')['retail_amount'].sum().sort_values(ascending=False)
    province_customers = df.groupby('province')['customer_id_number'].nunique().sort_values(ascending=False)
    province_metrics = df.groupby('province').agg(
        total_revenue=('retail_amount', 'sum'),
        unique_customers=('customer_id_number', 'nunique')
    )
    province_metrics['arpu'] = province_metrics['total_revenue'] / province_metrics['unique_customers']
    province_arpu = province_metrics['arpu'].sort_values(ascending=False)

    # Top 5 products overall
    top_products = df['product_name'].value_counts().head(5).index.tolist()
    province_product_pct = (
        pd.crosstab(df['province'], df['product_name'], normalize='index') * 100
    )
    has_all_top = all(prod in province_product_pct.columns for prod in top_products)

    # Build subplots: 2 rows × 2 cols
    specs = [
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "heatmap" if has_all_top else "table"}]
    ]
    subplot_titles = [
        "Total Revenue by Province",
        "Number of Unique Customers by Province",
        "Average Revenue per User (ARPU) by Province",
        "Product Preference by Province" if has_all_top else "Top Product by Province"
    ]
    fig = make_subplots(
        rows=2, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )

    # -------------------------------------------------------------------------
    # (1) Row 1, Col 1 → Total Revenue by Province (horizontal bar)
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=province_revenue.values,
            y=province_revenue.index.tolist(),
            orientation='h',
            marker_color=oldmutual_palette[0],
            name="Revenue",
            hovertemplate="Province: %{y}<br>Revenue: R%{x:,.0f}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Revenue (R)",
        tickprefix="R",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Province",
        row=1, col=1
    )

    # -------------------------------------------------------------------------
    # (2) Row 1, Col 2 → Unique Customers by Province (horizontal bar)
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=province_customers.values,
            y=province_customers.index.tolist(),
            orientation='h',
            marker_color=oldmutual_palette[1],
            name="Unique Customers",
            hovertemplate="Province: %{y}<br>Customers: %{x:,}<extra></extra>"
        ),
        row=1, col=2
    )
    fig.update_xaxes(
        title_text="Number of Customers",
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Province",
        row=1, col=2
    )

    # -------------------------------------------------------------------------
    # (3) Row 2, Col 1 → ARPU by Province (horizontal bar)
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=province_arpu.values,
            y=province_arpu.index.tolist(),
            orientation='h',
            marker_color=oldmutual_palette[2],
            name="ARPU",
            hovertemplate="Province: %{y}<br>ARPU: R%{x:.2f}<extra></extra>"
        ),
        row=2, col=1
    )
    fig.update_xaxes(
        title_text="ARPU (R)",
        tickprefix="R",
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Province",
        row=2, col=1
    )

    # -------------------------------------------------------------------------
    # (4) Row 2, Col 2 → Product Preference by Province (heatmap) OR Top Product (table)
    # -------------------------------------------------------------------------
    if has_all_top:
        # Prepare heatmap z-values: provinces × top_products (%)
        heat_vals = province_product_pct[top_products].reindex(province_revenue.index).fillna(0).values
        fig.add_trace(
            go.Heatmap(
                z=heat_vals,
                x=top_products,
                y=province_revenue.index.tolist(),
                colorscale=oldmutual_palette,
                hovertemplate="Province: %{y}<br>Product: %{x}<br>%: %{z:.1f}%<extra></extra>",
                colorbar=dict(title="% Preference")
            ),
            row=2, col=2
        )
        fig.update_xaxes(
            title_text="Product",
            tickangle=45,
            row=2, col=2
        )
        fig.update_yaxes(
            title_text="Province",
            row=2, col=2
        )
    else:
        # Build DataFrame of top product per province
        top_by_province = df.groupby('province')['product_name'] \
                             .agg(lambda s: s.value_counts().idxmax()) \
                             .reset_index(name='top_product') \
                             .reindex(columns=['province','top_product'])
        # Build a simple table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Province", "Top Product"],
                    fill_color='lightgrey',
                    align='center',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[top_by_province['province'], top_by_province['top_product']],
                    fill_color=[['white']*len(top_by_province), ['#f9f9f9']*len(top_by_province)],
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=2, col=2
        )

    # =============================================================================
    # Final layout tweaks
    # =============================================================================
    fig.update_layout(
        height=850,
        width=1400,
        margin=dict(t=100, b=60, l=80, r=80),
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=False
    )

    return fig

def plot_customer_lifetime_cohort(df):
    """
    Create Plotly visualizations for customer lifetime value and cohort analysis.
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import optimize

    df = df.copy()

    # -----------------------------------------------------------------------------
    # 1) Build Cohort Data for Retention Heatmap
    # -----------------------------------------------------------------------------
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

    cohort_data = (
        df_cohort.groupby(['cohort_month', 'cohort_month_num'])['customer_id_number']
                 .nunique()
                 .reset_index(name='unique_customers')
    )
    cohort_pivot = cohort_data.pivot(
        index='cohort_month', columns='cohort_month_num', values='unique_customers'
    ).fillna(0)
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_rates = cohort_pivot.divide(cohort_sizes, axis=0).fillna(0)

    cohort_labels = [str(p) for p in retention_rates.index.tolist()]
    cohort_month_nums = retention_rates.columns.tolist()
    retention_matrix = retention_rates.values

    # -----------------------------------------------------------------------------
    # 2) Build Cumulative Revenue by Cohort
    # -----------------------------------------------------------------------------
    cohort_revenue = (
        df_cohort.groupby(['cohort_month', 'cohort_month_num'])['retail_amount']
                 .sum()
                 .reset_index(name='revenue')
    )
    revenue_pivot = cohort_revenue.pivot(
        index='cohort_month', columns='cohort_month_num', values='revenue'
    ).fillna(0)
    revenue_cumulative = revenue_pivot.cumsum(axis=1)

    last_col = revenue_cumulative.columns.max() if not revenue_cumulative.empty else 0
    if last_col in revenue_cumulative.columns:
        top_cohorts = revenue_cumulative[last_col].nlargest(5).index.tolist()
    else:
        top_cohorts = revenue_cumulative.index.tolist()

    cohort_month_nums_cum = revenue_cumulative.columns.tolist()

    # -----------------------------------------------------------------------------
    # 3) Build Tenure Bucket Metrics for Dual-Axis
    # -----------------------------------------------------------------------------
    df['acquisition_date'] = df.groupby('customer_id_number')['customer_created_at'].transform('min')
    df['days_since_acquisition'] = (df['created_at'] - df['acquisition_date']).dt.days
    tenure_bins = [-1, 7, 30, 90, 180, 365, float('inf')]
    tenure_labels = ['First Week', '8-30 Days', '31-90 Days', '91-180 Days', '181-365 Days', '365+ Days']
    df['tenure_bucket'] = pd.cut(df['days_since_acquisition'], bins=tenure_bins, labels=tenure_labels)

    tenure_metrics = (
        df.groupby('tenure_bucket')
          .agg(
              total_revenue=('retail_amount', 'sum'),
              transaction_count=('id', 'count'),
              avg_order_value=('retail_amount', 'mean'),
              customer_count=('customer_id_number', pd.Series.nunique)
          )
          .reset_index()
    )
    tenure_metrics['revenue_per_customer'] = (
        tenure_metrics['total_revenue'] / tenure_metrics['customer_count']
    ).fillna(0)
    tenure_x = list(range(len(tenure_labels)))
    tenure_y_revenue_pc = tenure_metrics['revenue_per_customer'].values
    tenure_y_avg_order = tenure_metrics['avg_order_value'].fillna(0).values

    # -----------------------------------------------------------------------------
    # 4) Build Customer Value Growth Curve
    # -----------------------------------------------------------------------------
    df['week_since_acquisition'] = (df['days_since_acquisition'] // 7) + 1
    max_weeks = 24
    weekly_spend = (
        df[df['week_since_acquisition'] <= max_weeks]
        .groupby('week_since_acquisition')['retail_amount']
        .sum()
        .sort_index()
    )
    total_customers = df['customer_id_number'].nunique() or 1
    weekly_avg_cum = (weekly_spend.cumsum() / total_customers).fillna(0)

    growth_project = None
    if len(weekly_avg_cum) > 5:
        def log_func(x, a, b, c): return a * np.log(b * x) + c
        x_data = weekly_avg_cum.index.values
        y_data = weekly_avg_cum.values
        try:
            popt, _ = optimize.curve_fit(log_func, x_data, y_data, maxfev=5000)
            a, b, c = popt
            proj_weeks = 52
            x_proj = np.arange(1, proj_weeks + 1)
            y_proj = log_func(x_proj, a, b, c)
            growth_project = (x_proj, y_proj)
        except:
            growth_project = None

    # -----------------------------------------------------------------------------
    # 5) Build CLV Model Text & Segment CLV Bar Data
    # -----------------------------------------------------------------------------
    avg_rev_per_order = df['retail_amount'].mean() or 0
    avg_orders_per_cust = df.groupby('customer_id_number').size().mean() or 0
    if retention_rates.shape[1] > 1:
        retention_rate = retention_rates.iloc[:, 1].mean()
    else:
        retention_rate = 0.5
    retention_rate = retention_rate if not np.isnan(retention_rate) else 0.5
    avg_lifespan = (1 / (1 - retention_rate)) if retention_rate < 1 else 36
    monthly_discount = 0.10 / 12
    orders_per_month = avg_orders_per_cust / 3
    simple_clv = avg_rev_per_order * orders_per_month * avg_lifespan
    margin = 0.3 * avg_rev_per_order * orders_per_month
    discounted_clv = margin * (retention_rate / (1 + monthly_discount - retention_rate))

    model_lines = [
        "Customer Lifetime Value Model",
        "-----------------------------",
        f"Average Revenue per Order: R{avg_rev_per_order:.2f}",
        f"Average Orders per Customer: {avg_orders_per_cust:.2f}",
        f"Estimated Monthly Retention Rate: {retention_rate:.1%}",
        f"Estimated Avg Customer Lifespan: {avg_lifespan:.1f} months",
        "Assumed Profit Margin: 30%",
        "Annual Discount Rate: 10%",
        "",
        "Simple CLV Calculation:",
        "------------------------",
        f"CLV = Avg Rev/Order × Orders/Month × Avg Lifespan",
        f"CLV = R{avg_rev_per_order:.2f} × {orders_per_month:.2f} × {avg_lifespan:.1f}",
        f"CLV = R{simple_clv:.2f}",
        "",
        "Discounted CLV Calculation:",
        "---------------------------",
        f"CLV = Margin × (Retention / (1 + Discount - Retention))",
        f"CLV = R{margin:.2f} × ({retention_rate:.1%} / (1 + {monthly_discount:.1%} - {retention_rate:.1%}))",
        f"CLV = R{discounted_clv:.2f}"
    ]
    model_text = "<br>".join(model_lines)

    if 'age_group' in df.columns:
        seg_field = 'age_group'
    else:
        cust_metrics = df.groupby('customer_id_number').agg(
            total_spent=('retail_amount','sum'),
            frequency=('id','count')
        )
        quantiles = cust_metrics['total_spent'].quantile([0.25,0.5,0.75])
        def seg_label(x):
            if x <= quantiles[0.25]:
                return 'Low Value'
            elif x <= quantiles[0.50]:
                return 'Med-Low Value'
            elif x <= quantiles[0.75]:
                return 'Med-High Value'
            else:
                return 'High Value'
        cust_metrics['segment'] = cust_metrics['total_spent'].apply(seg_label)
        df['value_segment'] = df['customer_id_number'].map(cust_metrics['segment'])
        seg_field = 'value_segment'

    seg_metrics = df.groupby(seg_field).agg(
        avg_order=('retail_amount','mean'),
        total_spent=('retail_amount','sum'),
        orders_per_cust=('id','count'),
        cust_count=('customer_id_number','nunique')
    ).reset_index()
    seg_metrics['orders_per_customer'] = seg_metrics['orders_per_cust'] / seg_metrics['cust_count']
    seg_metrics['orders_per_month'] = seg_metrics['orders_per_customer'] / 3
    seg_metrics['estimated_clv'] = seg_metrics['avg_order'] * seg_metrics['orders_per_month'] * avg_lifespan
    seg_metrics = seg_metrics.sort_values('estimated_clv', ascending=False)
    seg_names = seg_metrics[seg_field].tolist()
    seg_values = seg_metrics['estimated_clv'].tolist()

    # =============================================================================
    # Build Plotly Subplots: 3 rows × 2 cols, specifying which subplots support secondary_y
    # =============================================================================
    specs = [
        [{"type": "heatmap"}, {"type": "xy", "secondary_y": False}],
        [{"type": "xy",   "secondary_y": True}, {"type": "xy", "secondary_y": False}],
        [{"type": "xy",   "secondary_y": False}, {"type": "xy", "secondary_y": False}]
    ]
    subplot_titles = [
        "Monthly Cohort Retention Rate",
        "Cumulative Revenue by Cohort",
        "Revenue per Customer & Avg Order by Tenure",
        "Customer Value Growth Curve",
        "CLV Model Summary",
        "Estimated CLV by Segment"
    ]

    fig = make_subplots(
        rows=3, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.12,
        row_heights=[0.3, 0.3, 0.3]
    )

    # -------------------------------------------------------------------------
    # (1) Row 1, Col 1: Heatmap of Retention Rates
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Heatmap(
            z=retention_matrix,
            x=cohort_month_nums,
            y=cohort_labels,
            colorscale=oldmutual_palette,
            hovertemplate="Cohort: %{y}<br>Month %{x}: %{z:.0%}<extra></extra>",
            showscale=True,
            colorbar=dict(title="% Retention")
        ),
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Months Since Acquisition",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Acquisition Cohort",
        row=1, col=1
    )

    # -------------------------------------------------------------------------
    # (2) Row 1, Col 2: Cumulative Revenue by Cohort (line traces)
    # -------------------------------------------------------------------------
    for cohort in top_cohorts:
        fig.add_trace(
            go.Scatter(
                x=cohort_month_nums,
                y=revenue_cumulative.loc[cohort].values,
                mode='lines+markers',
                line=dict(width=2),
                name=str(cohort),
                hovertemplate="Cohort: %{name}<br>Month %{x}: R%{y:,.0f}<extra></extra>"
            ),
            row=1, col=2
        )
    fig.update_xaxes(
        title_text="Months Since Acquisition",
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Cumulative Revenue (R)",
        row=1, col=2,
        tickprefix="R"
    )

    # -------------------------------------------------------------------------
    # (3) Row 2, Col 1: Revenue per Customer (bar) & Avg Order Value (line)
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=tenure_x,
            y=tenure_y_revenue_pc,
            marker_color=oldmutual_palette[0],
            name="Revenue per Customer",
            hovertemplate="Tenure: %{x}<br>R%{y:.2f}<extra></extra>"
        ),
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=tenure_x,
            y=tenure_y_avg_order,
            mode='lines+markers',
            line=dict(color=oldmutual_palette[3], width=2),
            name="Avg Order Value (R)",
            hovertemplate="Tenure: %{x}<br>R%{y:.2f}<extra></extra>"
        ),
        row=2, col=1, secondary_y=True
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tenure_x,
        ticktext=tenure_labels,
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Revenue per Customer (R)",
        row=2, col=1,
        secondary_y=False,
        tickprefix="R"
    )
    fig.update_yaxes(
        title_text="Avg Order Value (R)",
        row=2, col=1,
        secondary_y=True,
        tickprefix="R"
    )

    # -------------------------------------------------------------------------
    # (4) Row 2, Col 2: Customer Value Growth Curve
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=weekly_avg_cum.index,
            y=weekly_avg_cum.values,
            mode='lines+markers',
            line=dict(color=oldmutual_palette[1], width=2),
            name="Observed Cumulative Value",
            hovertemplate="Week %{x}: R%{y:,.2f}<extra></extra>"
        ),
        row=2, col=2
    )
    if growth_project is not None:
        x_proj, y_proj = growth_project
        fig.add_trace(
            go.Scatter(
                x=x_proj[x_proj > max_weeks],
                y=y_proj[x_proj > max_weeks],
                mode='lines',
                line=dict(color=oldmutual_palette[1], dash='dash', width=2),
                name="Projected Growth",
                hovertemplate="Week %{x}: R%{y:,.2f}<extra></extra>"
            ),
            row=2, col=2
        )
    fig.update_xaxes(
        title_text="Weeks Since Acquisition",
        row=2, col=2
    )
    fig.update_yaxes(
        title_text="Cumulative Value per Customer (R)",
        row=2, col=2,
        tickprefix="R"
    )

    # -------------------------------------------------------------------------
    # (5) Row 3, Col 1: CLV Model Text as an Annotation
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(x=[None], y=[None], showlegend=False),
        row=3, col=1
    )
    fig.add_annotation(
        text=model_text,
        xref="x domain", yref="y domain",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(family="Arial", size=12),
        align="left",
        xanchor="center",
        yanchor="middle",
        row=3, col=1
    )

    # -------------------------------------------------------------------------
    # (6) Row 3, Col 2: Estimated CLV by Segment (horizontal bar)
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=seg_values,
            y=seg_names,
            orientation='h',
            marker_color=oldmutual_palette[: len(seg_names)],
            name="Segment CLV",
            hovertemplate="Segment: %{y}<br>CLV: R%{x:,.2f}<extra></extra>"
        ),
        row=3, col=2
    )
    fig.update_xaxes(
        title_text="Customer Lifetime Value (R)",
        tickprefix="R",
        row=3, col=2
    )
    fig.update_yaxes(
        title_text=seg_field.replace('_', ' ').title(),
        row=3, col=2
    )

    # =============================================================================
    # Final layout tweaks
    # =============================================================================
    fig.update_layout(
        height=1000,
        width=1400,
        margin=dict(t=100, b=60, l=80, r=80),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        font=dict(family="Arial, sans-serif", size=12)
    )

    return fig



# ----------------------------------------
# Sidebar: Title & Filters
# ----------------------------------------
st.sidebar.title("⟡ Order Data Filters ⟡")
st.sidebar.markdown("Use the controls below to filter data across all tabs.\n")

# 1) Payment-Type Tabs
tab = st.sidebar.radio(
    "Select a View:", 
    ["Standard Payments","SIM Provisioning", "Reward", "Cost Centre", "Raw","Airtime"],
    index=0
)

# 2) Date Range Filter
# Compute min/max on Timestamp level first to avoid comparing date vs float
min_date = df["created_at"].dropna().min().date()
max_date = df["created_at"].dropna().max().date()

start_date, end_date = st.sidebar.date_input(
    "Date Range:",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)

# Apply the date filter globally
date_mask = (
    (df["created_at"].dt.date >= start_date) & 
    (df["created_at"].dt.date <= end_date)
)
filtered_df = df.loc[date_mask].copy()

# ----------------------------------------
# Payment-Type Mapping
# ----------------------------------------
payment_type_map = {
    "SIM Provisioning": ["SIM_PROVISIONING"],
    "Reward": ["REWARD"],
    "Cost Centre": ["COSTCENTRE"],
    "Standard Payments": [
        "EXT_VOUCHER", "POS", "VOUCHER", 
        "CUSTOMER_CC"
    ],
    "Airtime":["AIRTIME"],
}

# Airtime conversions 

# ----------------------------------------
# Main Content Area
# ----------------------------------------
st.title("✦ Order Data Dashboard ✦")
st.markdown(
    """
    *In this living canvas of data, select a tab on the left, choose a date range, 
    and watch insights unfold across each payment type.*
    """
)

if tab == "Raw":
    st.header("Raw Data Table")
    st.dataframe(filtered_df, use_container_width=True)

else:
    st.header(f"{tab} Insights")

       # Filter by payment type
    valid_types = payment_type_map.get(tab, [])
    df_tab = filtered_df[df["payment_type_name"].isin(valid_types)].copy()

    # 1) Weekly KPIs
    count = df_tab.shape[0]
    st.subheader(f"Data Points: {count:,}")
    st.write(f"Visualizations for **{tab}**, between **{start_date}** and **{end_date}**.")

    # Description for Weekly KPIs
    st.markdown("""
        **Description**:  
        This chart panel shows six subplots:
        1. **Weekly Order Volume** (lines+markers): total number of orders each week (resampled to Sundays).  
        2. **Weekly Revenue** (lines+markers): sum of `retail_amount` each week.  
        3. **Weekly Unique Customers** (lines+markers): count of distinct `customer_id_number` each week.  
        4. **Weekly Order Growth (%)** (bars): percent change in order volume from the previous week.  
        5. **Weekly Revenue Growth (%)** (bars): percent change in revenue from the previous week.  
        6. **Weekly Customer Growth (%)** (bars): percent change in unique customers from the previous week.

        **How it was created**:  
        The dataset is grouped by each calendar week (ending on Sunday) to compute totals for order count, revenue, and unique customers. From those weekly totals, percentages are calculated to show week-over-week growth. Finally, all six metrics are plotted together in a 2×3 grid so you can compare volume, revenue, and customer trends side by side.
    """)
    with st.expander("View code summary"):
        st.markdown("""
        ```python
        # Resample orders by week:
        weekly_orders = df_tab.resample('W-SUN', on='created_at').size()
        weekly_revenue = df_tab.resample('W-SUN', on='created_at')['retail_amount'].sum()
        weekly_customers = df_tab.resample('W-SUN', on='created_at')['customer_id_number'].nunique()

        # Compute week-over-week percentage changes:
        order_growth = weekly_orders.pct_change() * 100
        revenue_growth = weekly_revenue.pct_change() * 100
        customer_growth = weekly_customers.pct_change() * 100

        # Build Plotly subplots in a 2×3 layout:
        fig = make_subplots(rows=2, cols=3, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=weekly_orders.index, y=weekly_orders.values, mode='lines+markers', name='Order Volume'), row=1, col=1)
        fig.add_trace(go.Scatter(x=weekly_revenue.index, y=weekly_revenue.values, mode='lines+markers', name='Revenue'), row=1, col=2)
        fig.add_trace(go.Scatter(x=weekly_customers.index, y=weekly_customers.values, mode='lines+markers', name='Unique Customers'), row=1, col=3)
        fig.add_trace(go.Bar(x=order_growth.index, y=order_growth.values, name='Order Growth (%)'), row=2, col=1)
        fig.add_trace(go.Bar(x=revenue_growth.index, y=revenue_growth.values, name='Revenue Growth (%)'), row=2, col=2)
        fig.add_trace(go.Bar(x=customer_growth.index, y=customer_growth.values, name='Customer Growth (%)'), row=2, col=3)
        fig.update_layout(height=600, showlegend=False)
        ```  
        """)
    fig1 = plot_weekly_kpis_plotly(df_tab)
    st.plotly_chart(fig1, use_container_width=True)

    # 2) Bundle Size Distribution (only for Standard Payments)
    if tab == "Standard Payments":

        # 2a) Bundle Size Distribution Analysis
        st.markdown("### Bundle Size Distribution Analysis")
        st.markdown("""
        **Description**:  
        A histogram showing how many orders fall into each *data bundle size* category (e.g., `<30MB`, `30MB`, `50MB`, …, `20GB`). Each bar’s height represents the count of orders in that size range.

        **How it was created**:  
        First, the raw bundle sizes (in bytes) are converted to megabytes or, if missing, extracted from a JSON field in `product_details`. Invalid or zero-sized entries are removed. The remaining sizes are grouped into predefined bins (such as `<30MB`, `30MB`, `50MB`, …, `20GB`), and the count of orders in each bin is calculated. Finally, those counts are plotted as bars to show the distribution of bundle usage.
        """)
        with st.expander("View code summary"):
            st.markdown("""
            ```python
            # Ensure a column `bundleSize_MB` exists:
            if 'bundleSize' in df_tab.columns:
                df_tab['bundleSize_MB'] = df_tab['bundleSize'] / (1024 * 1024)
            else:
                df_tab['bundleSize_MB'] = df_tab['product_details'].apply(lambda x: json.loads(x).get('bundleSize', 0)) / (1024 * 1024)

            # Drop invalid sizes:
            bundle_data = df_tab[df_tab['bundleSize_MB'] > 0].copy()

            # Define bins and labels:
            bins = [0, 30, 50, 100, 250, 500, 1024, 2048, 5120, 10240, 20480, float('inf')]
            labels = ['<30MB', '30MB', '50MB', '100MB', '250MB', '500MB', '1GB', '2GB', '5GB', '10GB', '20GB']
            bundle_data['size_category'] = pd.cut(bundle_data['bundleSize_MB'], bins=bins, labels=labels, right=False)

            # Count orders per category:
            size_counts = bundle_data['size_category'].value_counts().sort_index()

            # Plot as a simple bar chart:
            fig = go.Figure([go.Bar(x=size_counts.index, y=size_counts.values)])
            fig.update_layout(xaxis_title='Bundle Size Category', yaxis_title='Order Count')
            ```  
            """)
        fig2 = plot_bundle_size_distribution_plotly(df_tab)
        st.plotly_chart(fig2, use_container_width=True)

        # 2b) Comprehensive Bundle Size Analysis
        st.markdown("### Comprehensive Bundle Size Analysis")
        st.markdown("""
        **Description**:  
        A 2×2 grid of subplots that includes:
        1. **Data Bundle Size Distribution** (top‐left): same counts as above, annotated with exact numbers over each bar.  
        2. **Average Revenue by Bundle Size** (top‐right): shows the mean `retail_amount` per size category; point sizes reflect the count of orders.  
        3. **Bundle Size Preference by Age Group** (bottom‐left): a heatmap showing, for the top age groups, what percentage of their orders falls into each size category.  
        4. **Bundle Size Popularity Over Time** (bottom‐right): a line plot of monthly order counts for the top 5 bundle categories, illustrating temporal trends.

        **How it was created**:  
        After categorizing each order by its size (as above), four separate summaries are generated:
        1. Order counts per size category, with values annotated above bars.
        2. Average revenue per category, computed by grouping orders by size and averaging `retail_amount`; order counts are used to scale marker sizes.
        3. If an `age_group` column exists: determine the top age brackets by order volume, then compute—for those brackets—the percentage of orders in each size bin, yielding a heatmap of preferences.
        4. Convert each order’s date to the first day of its month, then tally counts by size category per month; select the top 5 categories overall and plot their monthly counts.
        All four result sets are laid out in a 2×2 Plotly figure with consistent styling.
        """)
        with st.expander("View code summary"):
            st.markdown("""
            ```python
            # A: Distribution counts (same as before):
            size_counts = bundle_data['size_category'].value_counts().sort_index()

            # B: Average revenue and counts per category:
            size_revenue = bundle_data.groupby('size_category')['retail_amount'].agg(['mean','count']).reset_index()

            # C: Preference by age group (if present):
            if 'age_group' in bundle_data.columns:
                top_age_groups = bundle_data['age_group'].value_counts().head(5).index
                age_pref = (pd.crosstab(
                    bundle_data[bundle_data['age_group'].isin(top_age_groups)]['age_group'],
                    bundle_data[bundle_data['age_group'].isin(top_age_groups)]['size_category'],
                    normalize='index'
                ) * 100)

            # D: Monthly counts for top 5 categories:
            bundle_data['month'] = bundle_data['created_at'].dt.to_period('M').dt.to_timestamp()
            monthly_ct = pd.crosstab(bundle_data['month'], bundle_data['size_category'])
            top5_bundles = size_counts.nlargest(5).index
            monthly_top5 = monthly_ct[top5_bundles]
            monthly_top5 = monthly_top5.reset_index()

            # Build a 2×2 Plotly figure combining all four subplots.
            ```  
            """)
        fig3 = plot_bundle_size_analysis(df_tab)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("----")

        # 2c) Customer Lifecycle Value Analysis
        st.markdown("### Customer Lifecycle Value Analysis")
        st.markdown("""
        **Description**:  
        A 3×2 grid showing how customer-related metrics evolve by “lifecycle stage” (based on days since their first purchase). It includes:
        1. **Average Order Value by Lifecycle Stage** (bar): mean `retail_amount` for orders placed by customers in each stage.
        2. **Orders per Customer by Lifecycle Stage** (bar): average number of orders per customer in each stage.
        3. **Cumulative Revenue by Customer Tenure** (line): total revenue summed week by week since acquisition.
        4. **Product Category Preference by Lifecycle Stage** (heatmap): for the top 5 product categories, percentage of orders in each category by stage.
        5. **Average Bundle Size by Lifecycle Stage** (bar): if `bundleSize_MB` exists, mean bundle size per stage; otherwise, default to average order amount in the first month.

        **How it was created**:  
        Each customer’s first purchase date is identified and subtracted from each order’s date to compute “days since acquisition.” Those day counts are binned into six lifecycle stages (e.g., first week, 8–30 days, etc.). Metrics are then aggregated by stage:
        – Average order value and orders per customer are computed directly from `retail_amount` and order counts.
        – Weekly revenue is summed across all customers by “weeks since acquisition,” then cumulatively summed to show revenue buildup.
        – Product categories (extracted from `product_name`) are filtered to the top 5 categories overall, and a percentage cross-tabulation by stage yields a heatmap of preferences.
        – If data bundle sizes exist, mean bundle size per stage is calculated; otherwise, average order amount in the first 30 days is shown.
        All of these results are arranged into a 3×2 Plotly layout with bars, lines, and a heatmap.
        """)
        with st.expander("View code summary"):
            st.markdown("""
            ```python
            # Compute acquisition date and days since:
            acquisition = df_tab.groupby('customer_id_number')['customer_created_at'].min().reset_index()
            acquisition.columns = ['customer_id_number', 'acquisition_date']
            df_tab = df_tab.merge(acquisition, on='customer_id_number', how='left')
            df_tab['days_since_acquisition'] = (df_tab['created_at'] - df_tab['acquisition_date']).dt.days

            # Define lifecycle stages:
            bins = [-1, 7, 30, 90, 180, 365, float('inf')]
            labels = ['First Week', '8-30 Days', '31-90 Days', '91-180 Days', '181-365 Days', '365+ Days']
            df_tab['lifecycle_stage'] = pd.cut(df_tab['days_since_acquisition'], bins=bins, labels=labels)

            # A: Avg order value and orders per customer:
            stage_metrics = df_tab.groupby('lifecycle_stage').agg(
                avg_order_value=('retail_amount', 'mean'),
                num_orders=('id', 'count'),
                num_customers=('customer_id_number', pd.Series.nunique)
            ).reset_index()
            stage_metrics['orders_per_customer'] = stage_metrics['num_orders'] / stage_metrics['num_customers']

            # B: Weekly & cumulative revenue:
            df_tab['week_since_acquisition'] = (df_tab['days_since_acquisition'] // 7) + 1
            weekly_rev = df_tab.groupby('week_since_acquisition')['retail_amount'].sum()
            cumulative_rev = weekly_rev.cumsum()

            # C: Product category preference:
            df_tab['product_category'] = df_tab['product_name'].str.split().str[0]
            top5_cat = df_tab['product_category'].value_counts().head(5).index
            cat_stage = (pd.crosstab(
                df_tab[df_tab['product_category'].isin(top5_cat)]['lifecycle_stage'],
                df_tab[df_tab['product_category'].isin(top5_cat)]['product_category'],
                normalize='index'
            ) * 100)

            # D: Average bundle size by stage (if present):
            if 'bundleSize_MB' in df_tab.columns:
                bundle_stage = df_tab.groupby('lifecycle_stage')['bundleSize_MB'].mean().reset_index()
            else:
                bundle_stage = df_tab[df_tab['days_since_acquisition'] <= 30].groupby('lifecycle_stage')['retail_amount'].mean().reset_index()

            # Combine in a Plotly 3×2 figure.
            ```  
            """)
        fig4 = plot_customer_lifecycle_value(df_tab)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("----")

    else:
        st.info("Bundle size analysis only applicable for Standard Payments.")

    # 3) Temporal Patterns & Seasonality Analysis
    st.markdown("### Temporal Patterns & Seasonality Analysis")
    st.markdown("""
    **Description**:  
    This panel (split across multiple subplots) visualizes time-based order behavior:
    1. **Weekly Orders & Revenue** (bar + line on secondary y): total orders each week versus total revenue each week.
    2. **Weekly Customers & Avg Order Value** (bar + line): unique customers per week versus average order value per week.
    3. **Orders & Revenue by Day of Week** (bar + line): total orders versus revenue for each day of the week.
    4. **Orders & Avg Value by Hour of Day** (bar + line): distribution of order counts versus average order value for each hour (0–23).
    5. **Order Volume by Day of Month** (bar): total orders for each day in a calendar month.
    6. **Month-over-Month Growth** (line + markers): percent change in total monthly revenue from the previous month.

    **How it was created**:  
    Data is grouped by various time slices:
    – Weekly metrics are computed by aggregating orders, revenue, and unique customers per calendar week (Sunday-ending). Average order value is derived by dividing revenue by order count each week.
    – Day-of-week and hour-of-day metrics are calculated by extracting those fields from each order’s timestamp, then counting orders and averaging order value per bucket.
    – Day-of-month totals come from grouping by the day number (1–31) across all orders.
    – Month-over-month revenue growth is computed by summing revenue per calendar month and then calculating percentage change relative to the previous month.
    All these series are arranged in a 3×2 Plotly grid with shared styling, dual y-axes where needed.
    """)
    with st.expander("View code summary"):
        st.markdown("""
        ```python
        # Weekly metrics:
        weekly_orders = df_tab.resample('W-SUN', on='created_at').size()
        weekly_revenue = df_tab.resample('W-SUN', on='created_at')['retail_amount'].sum()
        weekly_customers = df_tab.resample('W-SUN', on='created_at')['customer_id_number'].nunique()
        weekly_avg_order = weekly_revenue / weekly_orders

        # Day of week and hour of day:
        df_tab['day_of_week'] = df_tab['created_at'].dt.day_name()
        df_tab['hour_of_day'] = df_tab['created_at'].dt.hour
        dow_orders = df_tab.groupby('day_of_week').size()
        dow_revenue = df_tab.groupby('day_of_week')['retail_amount'].sum()
        hod_orders = df_tab.groupby('hour_of_day').size()
        hod_avg_order = df_tab.groupby('hour_of_day')['retail_amount'].mean()

        # Day of month:
        df_tab['day_of_month'] = df_tab['created_at'].dt.day
        dom_counts = df_tab.groupby('day_of_month').size()

        # Month-over-month growth:
        mo_monthly = df_tab.resample('M', on='created_at')['retail_amount'].sum().reset_index(name='revenue')
        mo_monthly['revenue_pct'] = mo_monthly['revenue'].pct_change() * 100

        # Assemble Plotly subplots in a 3×2 layout.
        ```  
        """)
    fig6 = plot_temporal_patterns(df_tab)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("----")

    # 4) Geographic Analysis
    st.markdown("### Geographic Analysis")
    st.markdown("""
    **Description**:  
    A set of spatial and tabular plots showing how order metrics vary by geographic region. Typical subplots include:
    – A choropleth map of total orders or revenue by municipality or province.
    – A bar chart of top products or top revenue-generating regions.
    – A table summarizing, for each province, the product with the highest number of orders.

    **How it was created**:  
    Orders are grouped by a geographic field (e.g., `province` or `municipality`) to compute order counts and total revenue per region. The most-ordered product in each region is determined by finding the mode of `product_name` within that region. A choropleth (if geojson data is available) or bar charts and tables then visualize these aggregates.
    """)
    with st.expander("View code summary"):
        st.markdown("""
        ```python
        # Group by region:
        region_counts = df_tab.groupby('province').size().reset_index(name='order_count')
        region_revenue = df_tab.groupby('province')['retail_amount'].sum().reset_index(name='total_revenue')

        # Determine top product per province:
        top_by_province = (
            df_tab.groupby('province')['product_name']
                  .agg(lambda s: s.value_counts().idxmax())
                  .reset_index(name='top_product')
        )

        # Example: create a choropleth if geojson is present:
        fig = go.Figure(go.Choropleth(
            geojson=province_geojson,
            locations=region_counts['province'],
            z=region_counts['order_count'],
            featureidkey='properties.province_name'
        ))
        fig.update_layout(geo=dict(scope='south africa'))
        ```  
        """)
    fig7 = plot_geographic_analysis(df_tab)
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown("----")

    # 5) Cohort Analysis
    st.markdown("### Cohort Analysis")
    st.markdown("""
    **Description**:  
    This chart displays two main cohort-based visualizations:
    1. **Retention Heatmap**: for each monthly acquisition cohort, shows the percentage of that cohort remaining active in subsequent months.
    2. **Cumulative Revenue by Cohort**: shows how total revenue from each cohort accumulates over time.

    **How it was created**:  
    Each customer is assigned to an acquisition cohort based on the month of their first purchase. Then, every order’s date is mapped to its activity month, and the difference in months between an order’s activity month and the customer’s cohort month is computed to get the “cohort index.” Unique customer counts per cohort and period generate the retention matrix, which is normalized by the size of the initial cohort. Revenue sums per cohort and period produce a separate matrix. Both matrices are plotted as heatmaps to reveal retention and revenue patterns over time.
    """)
    with st.expander("View code summary"):
        st.markdown("""
        ```python
        def get_month_year(dt):
            return pd.Period(dt, freq='M')

        # Assign cohort month and activity month:
        df_tab['cohort_month'] = df_tab.groupby('customer_id_number')['customer_created_at'] \
                                      .transform('min').apply(get_month_year)
        df_tab['activity_month'] = df_tab['created_at'].apply(get_month_year)

        # Compute cohort index:
        df_tab['cohort_index'] = (df_tab['activity_month'].astype(int) - df_tab['cohort_month'].astype(int))

        # Build retention matrix:
        cohort_data = (
            df_tab.groupby(['cohort_month', 'cohort_index'])['customer_id_number']
                  .nunique().reset_index(name='unique_customers')
        )
        cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='unique_customers').fillna(0)
        cohort_sizes = cohort_pivot.iloc[:, 0]
        retention_rates = (cohort_pivot.divide(cohort_sizes, axis=0) * 100).fillna(0)

        # Build revenue matrix:
        cohort_revenue = (
            df_tab.groupby(['cohort_month', 'cohort_index'])['retail_amount']
                  .sum().reset_index(name='revenue')
        )
        revenue_pivot = cohort_revenue.pivot(index='cohort_month', columns='cohort_index', values='revenue').fillna(0)

        # Plot two heatmaps in a shared figure.
        ```  
        """)
    fig8 = plot_customer_lifetime_cohort(df_tab)
    st.plotly_chart(fig8, use_container_width=True)


    # 3) Filtered Data Table
    st.markdown("----")
    st.subheader("Filtered Data Table")
    st.dataframe(df_tab, use_container_width=True)

    st.markdown("### ▶️ Basic Statistics & Data Quality Diagnostics")

    # 1. BASIC STATISTICS
    # --------------------

    # Numeric columns: describe()
    num_cols = df_tab.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        st.text("➤ Numeric columns summary:\n" + df_tab[num_cols].describe().to_string())
    else:
        st.text("➤ No numeric columns to summarize.")

    # Categorical (object / bool / category) columns: value_counts (top 5 frequent)
    cat_cols = df_tab.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    if cat_cols:
        cat_summary = []
        for col in cat_cols:
            vc = df_tab[col].value_counts(dropna=False).head(5)
            cat_summary.append(f"{col!r} (top 5 categories):\n{vc.to_string()}")
        st.text("➤ Categorical columns summary:\n" + "\n\n".join(cat_summary))
    else:
        st.text("➤ No categorical columns to summarize.")

    # Datetime columns: min / max / count of uniques
    dt_cols = df_tab.select_dtypes(include=["datetime"]).columns.tolist()
    if dt_cols:
        dt_summary = []
        for col in dt_cols:
            series = df_tab[col].dropna()
            if not series.empty:
                dt_summary.append(
                    f"{col!r}:  min = {series.min()},  max = {series.max()},  unique = {series.nunique()}"
                )
            else:
                dt_summary.append(f"{col!r}:  (no non‐NA values)")
        st.text("➤ Datetime columns summary:\n" + "\n".join(dt_summary))
    else:
        st.text("➤ No datetime columns to summarize.")

    # 2. DATA QUALITY ISSUES / WARNINGS
    # ----------------------------------

    # 2a) Missing values: count & percentage
    missing_counts = df_tab.isna().sum()
    if missing_counts.sum() > 0:
        missing_pct = (missing_counts / len(df_tab) * 100).round(2)
        miss_df = pd.concat([missing_counts, missing_pct], axis=1, keys=["missing_count", "missing_pct"])
        miss_df = miss_df[miss_df["missing_count"] > 0]
        st.text("➤ Missing‐value report (columns with any NAs):\n" + miss_df.to_string())
    else:
        st.text("➤ No missing values detected.")

    # 2b) Duplicate rows: total count
    dup_count = df_tab.duplicated().sum()
    if dup_count > 0:
        st.text(f"➤ Duplicate‐row count: {dup_count}  (Consider dropping or reviewing these rows.)")
    else:
        st.text("➤ No duplicate rows found.")

    # 2c) Zero‐variance columns (all values identical)
    zero_var = [col for col in df_tab.columns if df_tab[col].nunique(dropna=False) <= 1]
    if zero_var:
        st.text("➤ Columns with zero variance (all identical):\n" + "\n".join(zero_var))
    else:
        st.text("➤ No zero‐variance columns detected.")


# ----------------------------------------
# Footer / Credits
# ----------------------------------------
st.markdown("""
---
© 2025 | Crafted with care for clarity, empathy, and forward-looking insights.
""")
