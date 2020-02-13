""" ETL main application entry point. Extract github data. Transform and clean
    relevant fields. Load into SQL database

    TODO:
        There is a balance between maximizing the output of a single query vs.
        how much of that query we want to store in RAM. 100 nodes worth of data
        (~ 100 KB) seems like a reasonable amount to store in RAM for
        transformation process.
"""
import json
import arrow
import pandas as pd
from gitstar import config
from gitstar.ETL import gqlquery
from gitstar.ETL.gstransform import transform

# Load GitHub PERSONAL ACCESS TOKEN
PAT = config.PAT
# Repo creation start, end, and last pushed. Format
CREATED_START = arrow.get("2015-01-01")
CREATED_END = arrow.get("2016-01-01")
LAST_PUSHED = arrow.get("2019-01-01")


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
    # Construct graphql query response generator
    gql_generator = gqlquery.GitHubSearchQuery(PAT, maxitems=10).generator()
    raw_data = next(gql_generator)
    clean_data = transform(raw_data)
    print_json(raw_data)
    print_json(clean_data)

    # pd_data = pd.DataFrame(data=clean_data)
    # print_json(clean_data)
    # print_pd(pd_data)

    # print("[{}] ETL begin.".format(arrow.now()))
    # while True:
    #    try:
    #        # Iterate generator. Normalize nested fields
    #        clean_data = GitStarTransform(next(gql_generator)).transform()
    #        print_json(clean_data)
    #    except StopIteration:
    #        print(
    #            "[{}] Reached end of query response. ETL done.".format(
    #                arrow.now()
    #            )
    #        )
    #        break
    # with open("gql_search_queries/query_out_sample", mode='w') as tfile:
    #   tfile.write(json.dumps(gdata, indent=4))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
