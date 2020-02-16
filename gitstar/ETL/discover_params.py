"""
    Determine appropriate search paramaters to use as daily
    gql github query. Probing number of repos created per day
    while varying stargazers. Important due to 1000 node limitation
    from graphql.
"""
import json
import logging
import arrow
import pyodbc
import matplotlib.pyplot as plt
import numpy as np
from gitstar import config
from gitstar.ETL import gqlquery
from gitstar.ETL.gstransform import repocount

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
CREATED_START = arrow.get("2019-12-29")
CREATED_END = arrow.get("2020-01-01")
LAST_PUSHED = arrow.get("2020-01-01")
MAXITEMS = 1


# For debugging
def print_json(obj):
    """Serialize python object to json formatted str and print"""
    print(json.dumps(obj, indent=4))


def set_logger():
    """Intialize root logger here."""
    logging.basicConfig(
        filename="logs/ETL.log",
        filemode="w",  # will rewrite on each run
        level=logging.INFO,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
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


def gql_generator(c_start, stars=0):
    """Construct graphql query response generator based on repo creation date
    """
    gql_gen = gqlquery.GitHubSearchQuery(
        PAT,
        created_start=c_start,
        created_end=c_start.shift(days=+1),
        last_pushed=LAST_PUSHED,
        maxitems=MAXITEMS,
        stars=stars,
    ).generator()
    return gql_gen


def repo_rate(created_start, created_end, stars):
    """Determine repos created per day based on star criteria.
        Stars are greater than given int value.
    """
    gql_gen = gql_generator(created_start, stars=stars)
    delta = bool((created_end - created_start).total_seconds())
    day = created_start
    dates_repos = {"dates": [], "repos": []}
    while delta:
        # Iterate generator. No pagination required.
        repo_count = repocount(next(gql_gen))
        params = {
            "CreatedStart": day,
            "CreatedEnd": day.shift(days=+1),
            "RepoCount": repo_count,
        }
        logging.info(params)
        day = day.shift(days=+1)
        gql_gen = gql_generator(day, stars)
        delta = bool((created_end - day).total_seconds())
        dates_repos["dates"].append(params["CreatedStart"])
        dates_repos["repos"].append(params["RepoCount"])
    return dates_repos


def plot_repo_rate(x, y):
    fig, ax = plt.subplots()
    ax.plot(y, y)
    #    ax.xaxis.set_major_formatter(lambda x: x.format("YYY
    ax.set(xlabel="Dates", ylabel="Repo Count", title="Repo Rate")
    return fig, ax


def main():
    """Execute ETL process"""
    set_logger()
    for star in range(1):
        rdict = repo_rate(CREATED_START, CREATED_END, star)
        dates = np.array(rdict["dates"])
        repos = np.array(rdict["repos"])
        print(dates)
        print(repos)
        fig, ax = plot_repo_rate(dates, repos)
    fig.show()
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
