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
import config
import gqlquery
from gstransform import GitStarTransform

# PAT = "PERSONAL_ACCESS_TOKEN"
PAT = config.PAT


def json_str(obj):
    """Serialize python object to json formatted str"""
    return json.dumps(obj, indent=4)


def print_json(obj):
    """Prints nice json string"""
    print(json.dumps(obj, indent=4))


def main():
    """Test the class implementations"""

    gql_generator = gqlquery.GitHubSearchQuery(PAT, maxitems=1).generator()
    clean_data = GitStarTransform(next(gql_generator)).transform()
    print_json(clean_data)
    #print_json(transform(gdata))
    #
    # with open("gql_search_queries/query_out_sample", mode='w') as tfile:
    #   tfile.write(json.dumps(gdata, indent=4))

    # while True:
    # for _ in range(3):
    #     # Print until StopIteration generator exception
    #     print_json(next(github_generator))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
