"""Load list of dictionaries into pd dataframe. Send to SQL database."""

import json
import pyodbc
import arrow
from gitstar import config
from ...ETL import gqlquery
from ...ETL.gstransform import transform

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


def load():
    """execute tests"""
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
    cursor.fast_executemany = True
    # --------------------------#
    gql_generator = gqlquery.GitHubSearchQuery(PAT, maxitems=2).generator()
    i = 1
    while True:
        try:
            raw_data = next(gql_generator)
            if i == 1:
                print(raw_data["data"]["search"]["repositoryCount"])
            clean_data = transform(raw_data)
            value_list = (list(node.values()) for node in clean_data)
            # Execute many statements.
            cursor.executemany(config.INSERT_QUERY, value_list)
            print(STATUS_MSG.format(cursor.rowcount))
            cnxn.commit()
            print(i)
            i += 1
        except StopIteration:
            print(
                "[{}] Reached end of query response. ETL done.".format(
                    arrow.now()
                )
            )
            break

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
