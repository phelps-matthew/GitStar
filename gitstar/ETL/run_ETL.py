""" ETL main application entry point. Extract github data. Transform and clean
    relevant fields. Load into SQL database

    TODO:
        There is a balance between maximizing the output of a single query vs.
        how much of that query we want to store in RAM. 100 nodes worth of data
        (~ 100 KB) seems like a reasonable amount to store in RAM for
        transformation process.
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
# Repo creation start, end, and last pushed. Format
CREATED_START = arrow.get("2018-02-26")
CREATED_END = arrow.get("2019-01-01")
LAST_PUSHED = arrow.get("2020-01-01")
MAXITEMS = 50


def print_json(obj):
    """Serialize python object to json formatted str and print"""
    print(json.dumps(obj, indent=4))


def print_pd(df):
    """Print pandas dataframe object"""
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "max_colwidth", 6
    ):
        print(df)


def main():
    """Execute ETL process"""
    global CREATED_START
    # Intialize root logger here
    logging.basicConfig(
        filename="ETL.log",
        filemode="w",
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
        created_start=CREATED_START,
        created_end=CREATED_START.shift(days=+1),
        last_pushed=LAST_PUSHED,
        maxitems=MAXITEMS,
    ).generator()
    logging.info("-" * 50)
    logging.info(
        "Begin main(). Created start date:{}".format(
            CREATED_START.format("YYYY-MM-DD")
        )
    )
    # Loop until end date
    delta = bool((CREATED_END - CREATED_START).seconds)
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
            CREATED_START = CREATED_START.shift(days=+1)
            gql_generator = gqlquery.GitHubSearchQuery(
                PAT,
                created_start=CREATED_START,
                created_end=CREATED_START.shift(days=+1),
                last_pushed=LAST_PUSHED,
                maxitems=MAXITEMS,
            ).generator()
            delta = bool((CREATED_END - CREATED_START).seconds)
            logging.info(
                "Reached end of GitHub query response. "\
                "New created start date:{}"
                .format(CREATED_START.format("YYYY-MM-DD"))
            )
            logging.info("-" * 20)
    logging.info("Exit main()")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
