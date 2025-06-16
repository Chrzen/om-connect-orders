# config/config.py
import textwrap

TABLE_NAME = "APP_SCHEMA.ORDER_DATA_10062025"

oldmutual_palette = [
    "#00524C", "#006B54", "#1A8754", "#5AAA46",
    "#8CC63F", "#E6F2E2"
]
negative_color = "#d65f5f"

# Info text definitions with the latest updates
info_texts = {
    "kpis": textwrap.dedent("""
        **What it shows:**

        A daily view of our revenue performance and its underlying trend.

        - **Left Axis (Green Area):** The total revenue earned each day.
        
        - **Right Axis (Bars):** The daily growth trend of our revenue, based on a 7-day rolling average. Green bars indicate a positive trend, red bars indicate a negative trend.

        **Business Question:**

        How does our daily revenue fluctuate, and what is the underlying growth momentum? This helps separate daily noise from true business performance.
        """),

    "prod_type": textwrap.dedent("""
        **What it shows:**

        A breakdown of our daily order volume and its growth trend.

        - **Left Axis (Stacked Area Chart):** Shows the daily order volume, broken down by product type. The total height of the stacked area represents the total orders for that day.
        
        - **Right Axis (Bars):** The growth trend of total daily orders, based on a 7-day rolling average.

        **Business Question:**

        Which of our product categories are driving sales volume, and is our overall order count growing or shrinking?
        """),

    "weekly_kpis": textwrap.dedent("""
        **What it shows:**

        A high-level weekly report card for our key business metrics.

        **How to read it:**

        - **Top Row:** Shows the absolute total for **Orders**, **Revenue**, and **Unique Customers** each week.
        - **Bottom Row:** Shows the **Week-over-Week Growth**. Green means growth compared to the prior week; red indicates a decline.

        **Business Question:**

        Are we growing consistently week by week? This is a key indicator of our business's health and momentum.
        """),

    "propensity": textwrap.dedent("""
        **What it shows:**

        This reveals how quickly customers come back to make another purchase after buying a specific product. Each line represents a top-selling product.

        **How to read it:**

        A line that rises sharply means customers repurchase very quickly. For example, if the '1GB Data' line hits 50% at the 15-day mark, it means half of the customers who bought 1GB of data made their next purchase within 15 days.

        **Business Question:**

        Which products are best at creating loyal, repeat customers? This can inform our marketing and retention strategies.
        """),

    "growth": textwrap.dedent("""
        **What it shows:**

        A simple bar chart measuring our recent performance momentum.

        **How to read it:**

        It compares the last 7 days to the 7 days prior (Week-over-Week), and the last 30 days to the 30 days prior (Month-over-Month).

        **Business Question:**

        Are we currently in a growth or decline phase? This is our short-term health check.
        """),

    "sankey": textwrap.dedent("""
        **What it shows:**

        This visualizes the most common paths customers take through their first four purchases. The wider the path, the more customers followed that route.

        **How to read it:**

        Follow the paths from left to right to see what a customer who first bought 'Product A' is most likely to buy second, third, and so on.

        **Business Question:**

        Is there a 'golden path' of purchases that leads to high-value customers? What products are key gateways to encourage further spending?
        """),

    "bundle": textwrap.dedent("""
        **What it shows:**

        A deep-dive analysis of our specific product bundles.

        **How to read it:**

        - **Left Chart:** Compares how many times a bundle is sold (bars) versus how much revenue it generates (line).
        - **Right Chart:** Shows which bundles are most popular each week.

        **Business Question:**

        Which bundles are our workhorses (high volume) versus our racehorses (high revenue)? Are there popular but low-revenue bundles we could optimize?
        """),

    "lifecycle": textwrap.dedent("""
        **What it shows:**

        How a customer's value and behavior changes over their lifetime with us, from their first week to over a year.

        **How to read it:**

        It tracks key metrics like average spend per order and purchase frequency as customers become more tenured. The heatmap shows which products are most popular at each stage of their journey.

        **Business Question:**

        Are we successfully increasing the value of our customers over time? Do we need different marketing strategies for new versus loyal customers?
        """),

    "cohort": textwrap.dedent("""
        **What it shows:**

        This groups customers by the week they made their first purchase (a 'cohort') and tracks their total spending over time.

        **How to read it:**

        - **Left Chart:** Tracks the cumulative revenue from our top-performing weekly cohorts.
        - **Right Chart:** Shows the average spending per customer and uses this trend to **project their 1-Year Lifetime Value (LTV)**.

        **Business Question:**

        How much is a new customer acquired today likely to be worth to us over the next year? This is crucial for budgeting marketing spend and forecasting future revenue.
        """),

    "geo_demo": textwrap.dedent("""
        **What it shows:**

        A breakdown of our performance by customer location (Province) and Age Group.

        **How to read it:**

        - **Bars:** Show which segments generate the most **Total Revenue**.
        - **Line:** Shows which segments are most engaged, measured by **Average Orders per Customer**.

        **Business Question:**

        Where are our most valuable customers located, and which age groups should we focus on? This helps target marketing efforts effectively.
        """),

    "prod_corr": textwrap.dedent("""
        **What it shows:**

        This heatmap reveals which products are most frequently bought together by the same customer.

        **How to read it:**

        The darker the square, the stronger the connection. A dark square where 'Product A' and 'Product B' intersect means customers who buy 'A' are very likely to also buy 'B'.

        **Business Question:**

        What are our best cross-selling or bundling opportunities? For example, if many people who buy 'Voice Minutes' also buy 'WhatsApp Bundles', we can promote them as a package.
        """),

    "sunbursts": textwrap.dedent("""
        **What it shows:**

        An interactive, hierarchical view of our revenue streams. You can click on segments to drill down.

        **How to read it:**

        - **Left Chart:** Starts with Province, then breaks down into Product Category.
        - **Right Chart:** Starts with Age Group, then breaks down into Product Category.

        **Business Question:**

        Who are our most profitable customer segments and what, specifically, are they buying? (e.g., "How much revenue from Millennials in Gauteng comes from Data vs. Airtime?")
        """)
}

styling = """
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
        background-color: rgba(38, 39, 48, 0.9);
        color: #fff;
        text-align: left;
        font-style: normal;
        font-size: 15px;
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
"""