import snowflake.connector

try:
    conn = snowflake.connector.connect(
        user='CHRIS',
        password='EventHorizon27!',
        account='kotjzmx-oe33721',  # Updated format
        warehouse='COMPUTE_WH',
        database='STREAMLIT_DB',
        schema='APP_SCHEMA',
        role='ACCOUNTADMIN'
    )
    print("✅ Connection successful!")
    
    # Test a simple query
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ORDER_DATA LIMIT 10")
    result = cursor.fetchone()
    print(f"✅ Found {result[0]} rows in ORDER_DATA table")
    
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")