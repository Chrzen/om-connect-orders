# upload_to_snowflake.py
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import streamlit as st

# 1. Read your local Parquet
df_local = pd.read_parquet("orderData_clean.parquet")
df_local["created_at"] = pd.to_datetime(df_local["created_at"])

# 2. Open a Snowflake connection (using the same secrets.toml as your app)
conn = snowflake.connector.connect(
    user=st.secrets["snowflake"]["user"],
    password=st.secrets["snowflake"]["password"],
    account=st.secrets["snowflake"]["account"],
    warehouse=st.secrets["snowflake"]["warehouse"],
    database=st.secrets["snowflake"]["database"],
    schema=st.secrets["snowflake"]["schema"],
    role=st.secrets["snowflake"]["role"],
)

# 3. Ensure the table exists (DDL must match your DataFrame schema)
conn.cursor().execute("""
CREATE OR REPLACE TABLE APP_SCHEMA.ORDER_DATA (
  id STRING,
  created_at TIMESTAMP_NTZ,
  accounted_at TIMESTAMP_NTZ,
  order_type_name STRING,
  order_action_name STRING,
  retail_amount FLOAT,
  retailer_discounted_amount FLOAT,
  payment_type_name STRING,
  order_metadata STRING,
  customer_reference STRING,
  customer_created_at TIMESTAMP_NTZ,
  customer_updated_at TIMESTAMP_NTZ,
  customer_type_name STRING,
  customer_attributes STRING,
  customer_status_code STRING,
  customer_status STRING,
  customer_address3 STRING,
  customer_address4 STRING,
  customer_address5 STRING,
  customer_id_number STRING,
  customer_id_type STRING,
  retailer_name STRING,
  retailer_type_name STRING,
  retailer_attributes STRING,
  retailer_status_code STRING,
  retailer_status STRING,
  product_name STRING,
  product_type_desc STRING,
  product_sku STRING,
  product_price_fixed FLOAT,
  product_price_min FLOAT,
  product_price_max FLOAT,
  product_details STRING,
  order_item_id STRING,
  order_item_created_at TIMESTAMP_NTZ,
  order_item_accounted_at TIMESTAMP_NTZ,
  order_item_product_id STRING,
  order_item_qty INTEGER,
  order_item_retail_amount FLOAT,
  order_item_provision_type_name STRING,
  order_item_provisioned_units FLOAT,
  order_item_provisioned_unit_type_name STRING,
  birth_year FLOAT,
  customer_age INTEGER,
  age_group STRING,
  province STRING,
  ext_voucher_msisdn STRING,
  ext_voucher_serial STRING,
  ext_voucher_source_irn STRING,
  sim_imsi STRING,
  sim_iccid STRING,
  voucher_serial STRING,
  voucher_legacy_txn_irn STRING,
  bundleSize FLOAT,
  serviceType STRING,
  payment_cohort STRING
);
""")

# 4. Bulk-upload with write_pandas
success, nchunks, nrows, _ = write_pandas(
    conn,
    df_local,
    "ORDER_DATA",          # table name
    database="STREAMLIT_DB",
    schema="APP_SCHEMA",
    chunk_size=50000       # adjust chunk size if needed
)
if success:
    print(f"✅ Uploaded {nrows} rows in {nchunks} chunks to APP_SCHEMA.ORDER_DATA")
else:
    print("❌ Upload failed. Check logs.")

conn.close()
