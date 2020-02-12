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
import config
from . import gqlquery
from .gstransform import transform

# Load GitHub PERSONAL ACCESS TOKEN
PAT = config.PAT


def print_json(obj):
    """Serialize python object to json formatted str and print"""
    print(json.dumps(obj, indent=4))


def print_pd(df):
    """Print pandas dataframe object"""
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "max_colwidth", 6
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
    # fmt: off
    import ipdb,os; ipdb.set_trace(context=5)  # noqa
    # fmt: on

    #pd_data = pd.DataFrame(data=clean_data)
    #print_json(clean_data)
    #print_pd(pd_data)

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
