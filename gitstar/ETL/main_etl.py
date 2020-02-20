""" ETL main application entry point. Extract github data. Transform and clean
    relevant fields. Load into SQL database. Fetch -> clean -> load process is
    iterated serially.

    Notes:
        GitHub limits had to be probed experimentally - did not adhere to
        rate limits as suggested in documentation. May only return 1000 repo
        nodes per search query.

        Repo data is iterated based on creation date, incremented daily
        through a range (c.f. created_start, created_end).

        50 items/(http request) was reasonable fetching param. Approx. 1KB
        data/repo to be held in RAM.

    TODO:
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
CREATED_START = arrow.get("2018-09-21")
CREATED_END = arrow.get("2019-12-31")
PUSH_START = arrow.get("2020-01-01")
MAXITEMS = 50
MINSTARS = 1
MAXSTARS = None


# For debugging
def print_json(obj):
    """Serialize python object to json formatted str and print"""
    print(json.dumps(obj, indent=4))


def set_logger(filename):
    """Intialize root logger here."""
    logging.basicConfig(
        filename=filename,
        filemode="w",  # will rewrite on each run
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )


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
        + PASSWORD
    )
    cursor = cnxn.cursor()
    # Combine INSERT's into single query
    cursor.fast_executemany = True
    return cursor


def dbload(odbc_cnxn, value_list):
    """Load columns into sql db"""
    odbc_cnxn.executemany(INSERT_QUERY, value_list)
    logging.info(STATUS_MSG.format(odbc_cnxn.rowcount))
    # Send SQL query to db
    odbc_cnxn.commit()


def gql_generator(c_start, pushed_start, pushed_end=None):
    """Construct graphql query response generator based on repo creation date.
        Date range {}..{} is inclusive.
    """
    gql_gen = gqlquery.GitHubSearchQuery(
        PAT,
        created_start=c_start,
        created_end=c_start,
        pushed_start=pushed_start,
        pushed_end=pushed_end,
        maxitems=MAXITEMS,
        minstars=MINSTARS,
        maxstars=MAXSTARS,
    ).generator()
    return gql_gen


def etl_loop(created_start, created_end, pushed_start, pushed_end=None):
    """Iterate through gql_generator (which handles pagination) while looping
        through provided date range. Raise repo count exception as needed.
    """
    # Intialize db connection and gql response
    dbcnxn = dbconnection()
    gql_gen = gql_generator(created_start, pushed_start, pushed_end=pushed_end)
    logging.info(
        "-" * 80
        + "Begin etl_loop(). Created start date:{}".format(
            created_start.format("YYYY-MM-DD")
        )
    )
    # Loop until end date
    delta = (created_end - created_start).total_seconds()
    day = created_start
    while delta >= 0:
        try:
            # Iterate generator. Normalize nested fields.
            clean_data = transform(next(gql_gen))
            # Construct generator of dict values
            value_list = (list(node.values()) for node in clean_data)
            # repos = [node["nameWithOwner"] for node in clean_data]
            # logging.info('\n'+'\n'.join(repos))
            # Load into db
            dbload(dbcnxn, value_list)
            print("[{}] {} rows inserted into db".format(arrow.now(), MAXITEMS))
        except (StopIteration, gqlquery.RepoCountError):
            day = day.shift(days=+1)
            gql_gen = gql_generator(
                day, pushed_start=pushed_start, pushed_end=pushed_end
            )
            delta = (created_end - day).total_seconds()
            logging.info(
                "Reached end of gql pagination or exceeded repo count. "
                "New created start date:{}\n".format(day.format("YYYY-MM-DD"))
                + "-" * 80
            )


def special_etl():
    """Dates that exceeded 1000 repo count must be treated specially
        Last pushed date range is sliced to fetch < 1000 repos/query
    """
    # Determined by logs of primary ETL
    special_dates = [
        ("2019-11-18", "2019-11-21"),
        ("2019-11-25", "2019-11-27"),
        ("2019-12-02", "2019-12-06"),
        ("2019-12-09", "2019-12-13"),
        ("2019-12-16", "2019-12-24"),
        ("2019-12-26", "2019-12-31"),
    ]
    # This range of push dates was verfied to return < 1k repos
    push_start1 = arrow.get("2020-01-01")
    push_end1 = arrow.get("2020-01-20")
    push_start2 = arrow.get("2020-01-21")
    for c_start, c_end in special_dates:
        # First push slice
        etl_loop(
            created_start=arrow.get(c_start),
            created_end=arrow.get(c_end),
            pushed_start=push_start1,
            pushed_end=push_end1,
        )
        # Second push slice
        etl_loop(
            created_start=arrow.get(c_start),
            created_end=arrow.get(c_end),
            pushed_start=push_start2,
        )


def main():
    """Execute ETL process"""
    set_logger("logs/ETL_special.log")

    # Primary ETL (uncomment below)
    etl_loop(CREATED_START, CREATED_END, PUSH_START)

    # Follow up ETL
    special_etl()

    logging.info("Exit main()")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
