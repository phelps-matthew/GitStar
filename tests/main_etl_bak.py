""" ETL main application entry point. Extract github data. Transform and clean
    relevant fields. Load into SQL database. Fetch -> clean -> load process is
    iterated serially.

    Notes:
        GitHub limits had to be probed experimentally - did not adhere to
        rate limits as suggested in documentation.

        Repo data is iterated based on creation date, incremented daily
        through a range (c.f. created_start, created_end).

        50 items/(http request) was reasonable fetching param. Approx. 1KB
        data/repo to be held in RAM.

    TODO:
        a
"""
import json
import logging
import arrow
import pyodbc
from gitstar import config
from gitstar.ETL import gqlquery
from gitstar.ETL.gstransform import transform

# GitHub PERSONAL ACCESS TOKEN
PAT = config.PAT
# SQL db params
SERVER = config.SERVER
DATABASE = config.DATABASE
USERNAME = config.USERNAME
PASSWORD = config.PASSWORD
DRIVER = "{ODBC Driver 17 for SQL Server}"
STATUS_MSG = "Executed SQL query. Affected row(s):{}"
INSERT_QUERY = config.INSERT_QUERY
# Repo creation start, end, and last pushed. Format
created_start = arrow.get("2018-02-26")  # Not static
CREATED_END = arrow.get("2019-01-01")
LAST_PUSHED = arrow.get("2020-01-01")
MAXITEMS = 50


# For debugging
def print_json(obj):
    """Serialize python object to json formatted str and print"""
    print(json.dumps(obj, indent=4))


def dbconnection():
    # Initialize sql db connection
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
    # Combine INSERT's into single query
    cursor.fast_executemany = True
    return cursor

def dbload(odbc_cnxn, value_list):
    odbc_cnxn.executemany(INSERT_QUERY, value_list)
    logging.info(STATUS_MSG.format(odbc_cnxn.rowcount))
    # Send SQL query to db
    odbc_cnxn.commit()


def set_logger():
    # Intialize root logger here.
    logging.basicConfig(
        filename="ETL.log", 
        filemode="w",  # will rewrite on each run
        level=logging.DEBUG,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    )





def main():
    """Execute ETL process"""
    global created_start
    # Intialize root logger here.
    logging.basicConfig(
        filename="ETL.log", 
        filemode="w",  # will rewrite on each run
        level=logging.DEBUG,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    )
    # Initialize sql db connection
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
    # Combine INSERT's into single query
    cursor.fast_executemany = True
    # Construct graphql query response generator
    gql_generator = gqlquery.GitHubSearchQuery(
        PAT,
        created_start=created_start,
        created_end=created_start.shift(days=+1),
        last_pushed=LAST_PUSHED,
        maxitems=MAXITEMS,
    ).generator()
    logging.info("-" * 50)
    logging.info(
        "Begin main(). Created start date:{}".format(
            created_start.format("YYYY-MM-DD")
        )
    )
    # Loop until end date
    delta = bool((CREATED_END - created_start).seconds)
    while not delta:
        try:
            # Iterate generator. Normalize nested fields.
            clean_data = transform(next(gql_generator))
            # Construct generator of dict values
            value_list = (list(node.values()) for node in clean_data)
            # Load SQL insert query
            cursor.executemany(config.INSERT_QUERY, value_list)
            logging.info(STATUS_MSG.format(cursor.rowcount))
            # Send SQL query to db
            cnxn.commit()
            print("[{}] {} rows inserted into db".format(arrow.now(), MAXITEMS))
        except StopIteration:
            created_start = created_start.shift(days=+1)
            gql_generator = gqlquery.GitHubSearchQuery(
                PAT,
                created_start=created_start,
                created_end=created_start.shift(days=+1),
                last_pushed=LAST_PUSHED,
                maxitems=MAXITEMS,
            ).generator()
            delta = bool((CREATED_END - created_start).seconds)
            logging.info(
                "Reached end of GitHub query response. "\
                "New created start date:{}"
                .format(created_start.format("YYYY-MM-DD"))
            )
            logging.info("-" * 80)
    logging.info("Exit main()")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()