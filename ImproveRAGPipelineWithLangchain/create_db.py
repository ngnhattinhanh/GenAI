import pandas as pd
import sqlite3

# Load the CSV file into a DataFrame
df_asamamove_sales_data = pd.read_csv("./Finance/asamamove_sales_data.csv")
df_supermarket_sales_data = pd.read_csv("./Finance/superstore_sale_01.csv")
df_employee_sales_data = pd.read_csv("./Finance/superstore_sale_02.csv")

df_hr_comma_sep = pd.read_csv("./HR/HR_comma_sep.csv")
df_hr_employee = pd.read_csv("./HR/HR-Employee-Attrition.csv")
df_hr_datavset14 = pd.read_csv("./HR/HR_dataset_v14.csv")

# Normalize column names
def normalize_column_names(df):
    df.columns = (
        df.columns.str.strip()  # Remove leading/trailing whitespace
        .str.lower()  # Convert to lowercase
    )
    return df

df_asamamove_sales_data = normalize_column_names(df_asamamove_sales_data)
df_supermarket_sales_data = normalize_column_names(df_supermarket_sales_data)
df_employee_sales_data = normalize_column_names(df_employee_sales_data)
df_hr_comma_sep = normalize_column_names(df_hr_comma_sep)
df_hr_employee = normalize_column_names(df_hr_employee)
df_hr_datavset14 = normalize_column_names(df_hr_datavset14)

# Create a SQLite database and store the DataFrames as tables
hr_db_connection = sqlite3.connect("HR_data.db")
finance_db_connection = sqlite3.connect("Finance_data.db")

df_hr_comma_sep.to_sql("HR_comma_sep", hr_db_connection, if_exists="replace", index=False)
df_hr_employee.to_sql("HR_employee", hr_db_connection, if_exists="replace", index=False)
df_hr_datavset14.to_sql("HR_dataset_v14", hr_db_connection, if_exists="replace", index=False)

df_asamamove_sales_data.to_sql("AsamaMove_Sales", finance_db_connection, if_exists="replace", index=False)
df_supermarket_sales_data.to_sql("Supermarket_Sales", finance_db_connection, if_exists="replace", index=False)
df_employee_sales_data.to_sql("Employee_Sales", finance_db_connection, if_exists="replace", index=False)

# The commit statements are not necessary if you are only querying data.
# They can be removed as no changes are being made to the database.