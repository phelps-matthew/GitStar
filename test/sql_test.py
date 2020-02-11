"""Load list of dictionaries into pd dataframe. Send to SQL database."""

import pandas as pd
import pyodbc
import config

SERVER = config.SERVER
DATABASE = config.DATABASE
USERNAME = config.USERNAME
PASSWORD = config.PASSWORD
DRIVER = "{ODBC Driver 17 for SQL Server}"
cnxn = pyodbc.connect(
    "DRIVER="
    + DRIVER
    + ";SERVER="
    + SERVER
    + ";PORT=1433;DATABASE="
    + DATABASE
    + ";UID="
    + USERNAME
    + ";PWD="
    + PASSWORD
)
cursor = cnxn.cursor()
cursor.execute(
    "SELECT TOP 20 pc.Name as CategoryName, p.name as ProductName FROM [SalesLT].[ProductCategory] pc JOIN [SalesLT].[Product] p ON pc.productcategoryid = p.productcategoryid"
)
row = cursor.fetchone()
while row:
    print(str(row[0]) + " " + str(row[1]))
    row = cursor.fetchone()
