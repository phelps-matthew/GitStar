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
import matplotlib.dates as mdates
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
CREATED_START = arrow.get("2015-01-01")
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
        filename="logs/discover_params.log",
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


def gql_generator(c_start, minstars=0):
    """Construct graphql query response generator based on repo creation date
    """
    gql_gen = gqlquery.GitHubSearchQuery(
        PAT,
        created_start=c_start,
        created_end=c_start,
        last_pushed=LAST_PUSHED,
        maxitems=MAXITEMS,
        minstars=minstars,
    ).generator()
    return gql_gen


def repo_rate(created_start, created_end, minstars):
    """Determine repos created per day based on star criteria.
        Stars are greater than given int value.
    """
    gql_gen = gql_generator(created_start, minstars=minstars)
    delta = (created_end - created_start).total_seconds()
    day = created_start
    dates_repos = {"dates": [], "repos": []}
    while delta >= 0:
        # Iterate generator. No pagination required.
        repo_count = repocount(next(gql_gen))
        params = {
            "CreatedStart": day,
            "RepoCount": repo_count,
        }
        logging.info(params)
        dates_repos["dates"].append(day)
        dates_repos["repos"].append(repo_count)
        day = day.shift(days=+1)
        gql_gen = gql_generator(day, minstars=minstars)
        delta = (created_end - day).total_seconds()
    return dates_repos


def plot_repo_rate(x, y, stars):
    """Plot arrow date list (x) and repo count (y) with star criteria"""
    # Convert arrow -> datetime -> np.datetime64, store in np.array
    x = np.array(list(map(lambda z: np.datetime64(z.datetime), x)))
    y = np.array(y)
    # x axis date parsing
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter("%Y")
    # Create Figure.figure and Axes.axes instances
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, "b.")
    ax.set(
        ylabel="Repos Created (daily)",
        title="Created Repos, Stars:>{}".format(stars),
    )
    # Format x ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    # Add range to nearest month
    datemin = np.datetime64(x[0], "M") - np.timedelta64(2, "M")
    datemax = np.datetime64(x[-1], "M") + np.timedelta64(2, "M")
    ax.set_xlim(datemin, datemax)
    # Format coordinates
    ax.format_xdata = mdates.DateFormatter("%Y-%m-%d")
    ax.grid(True)
    # Rotate x labels, makes room
    fig.autofmt_xdate()
    return fig, ax


def star_write(rdict, star):
    # Convert arrow to str for json encoding
    rdict["dates"] = list(map(lambda x: x.format(), rdict["dates"]))
    with open("data/repo_star_{}".format(star), "w") as file:
        json.dump(rdict, file, indent=4)


def star_read(star):
    with open("data/repo_star_{}".format(star)) as file:
        rdict = json.load(file)
        return rdict


def main():
    """Execute ETL process"""
    set_logger()
    # star = 1
    # rdict = star_read(star)
    # # Convert from str to arrow
    # rdict["dates"] = list(map(lambda x: arrow.get(x), rdict["dates"]))
    # dates = rdict["dates"]
    # repos = rdict["repos"]
    # fig, ax = plot_repo_rate(dates, repos, star)
    # # plt.show()
    # fig.savefig(
    #     "data/repo_star_{}.png".format(star),
    #     transparent=False,
    #     dpi=100,
    #     bbox_inches="tight",  # fit bounds of figure to plot
    # )
    for star in range(3):
        rdict = repo_rate(CREATED_START, CREATED_END, star)
        star_write(rdict, star)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
