"""Load list of dictionaries into pd dataframe. Send to SQL database."""

import json
import pyodbc
import arrow
import gitstar.ETL
from gitstar import config
from gitstar.ETL import gqlquery
import pandas as pd
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent
FILENAME = "gs_table_v2.csv"
# db configs
SERVER = config.SERVER
DATABASE = config.DATABASE
USERNAME = config.USERNAME
PASSWORD = config.PASSWORD
DRIVER = "{ODBC Driver 17 for SQL Server}"
STATUS_MSG = "Executed SQL query. Affected row(s):{}"
# Load GitHub PERSONAL ACCESS TOKEN
PAT = config.PAT


def print_json(obj):
    """Serialize python object to json formatted str and print"""
    print(json.dumps(obj, indent=4))


def dbconnection():
    """Intialize sql db connection"""
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
        + PASSWORD,
        autocommit=True,
    )
    cursor = cnxn.cursor()
    cursor.fast_executemany = True
    return cnxn, cursor


def table_to_csv(sql, file_path):
    """
    This function creates csv file from the query result with ODBC driver
    """
    dbcnxn, _ = dbconnection()
    df = pd.read_sql(sql, dbcnxn)
    df.to_csv(
        file_path,
        encoding="utf-8",
        header=True,
        doublequote=True,
        sep=",",
        index=False,
    )
    dbcnxn.close()


sql_statement = "SELECT * FROM gs_table_v2"

table_to_csv(sql_statement, BASE_DIR / FILENAME)

# cursor.execute(
#    """
#    ALTER TABLE mytable_example
#    ALTER COLUMN
#    c_col DATETIME2(0);
# """,
# )
# print(STATUS_MSG.format(cursor.rowcount))
# cnxn.commit()
# cursor.execute(
#    """
#    INSERT INTO mytable_example
#    (a_col, b_col, c_col)
#    VALUES
#    (?, ?, ?)
# """,
#    (123, "asdf", "asdf2020-02-10T21:16:15Z"),
# )
# print(STATUS_MSG.format(cursor.rowcount))
# cnxn.commit()
#
#
# cursor.execute(
#    """
#    SELECT TOP 20 pc.Name as CategoryName, p.name as ProductName
#        FROM [SalesLT].[ProductCategory] pc JOIN [SalesLT].[Product] p
#        ON pc.productcategoryid = p.productcategoryid
# """
# )
# row = cursor.fetchone()
# while row:
#    print(str(row[0]) + " " + str(row[1]))
#    row = cursor.fetchone()
#
## pass parameters with question marks
# cursor.execute(
#    """
#    select user_id, user_name
#      from users
#     where last_logon < ?
#       and bill_overdue = ?
# """,
#    [datetime.date(2001, 1, 1), "y"],
# )
#
## Use commit on INSERT and CREATE. The query statement is kept, so repeat of execute() repeats last query
# cursor.execute(
#    "insert into products(id, name) values (?, ?)", "pyodbc", "awesome library"
# )
# cnxn.commit()
#
## Execute many statements.
# params = [("A", 1), ("B", 2)]
# cursor.fast_executemany = True
# cursor.executemany("insert into t(name, id) values (?, ?)", params)
