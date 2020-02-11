""" ETL main application entry point. Extract github data. Transform and clean relevant
fields. Load into SQL database

    TODO:
        There is a balance between maximizing the output of a single query vs. how much
        of that query we want to store in RAM. 100 nodes worth of data (~ 100 KB) seems
        like a reasonable amount to store in RAM for transformation process.
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


def normalize(ndict):
    """Take in single node (nested dictionary representing a single repo) and remove
        nested fields. Hard coded keylists for speed, as opposed to iterating through
        all keys. Less general, but faster.

        Returns single dictionary.
    """

    # Remove node header
    ndict = ndict["node"]
    # List of keys with nested dicts of depth=2
    dict_keys_d2 = [
        "readme",
        "primaryLanguage",
        "releases",
        "stargazers",
        "watchers",
        "deployments",
        "repositoryTopics",
        "pullRequests",
        "projects",
        "milestones",
        "issuelabels",
        "closedissues",
        "openissues",
    ]
    # List of keys with nested dicts of depth=3
    dict_keys_d3 = [("commitnum", "history")]

    # Two for loops are ugly, but again.. speed! We don't want comparisons on all
    # possible fields. Also faster to catch isolated exception
   
    # Restructure to depth=1.
    for key in dict_keys_d2:
        # Catch null values that would otherwise be nested dicts. Log null errors.
        try:
            value = ndict[key].popitem()[1]
            ndict[key] = value
        except AttributeError:
            logging.warning(
                "Null field found. Repository:{} key:{}".format(
                    ndict["nameWithOwner"], key
                )
            )
    # fmt: off
    import ipdb,os; ipdb.set_trace(context=5)  # noqa
    # fmt: on
    for key in dict_keys_d3:
        # Catch null values that would otherwise be nested dicts. Log null errors.
        try:
            value = ndict[key[0]][key[1]].popitem()[1]
            ndict[key[0]] = value
        except AttributeError:
            logging.warning(
                "Null field found. Repository:{} key:{}".format(
                    ndict["nameWithOwner"], key
                )
            )
    return ndict


def transformx(ndicts):
    """Normalizes json decoded graphql data (a nested dictionary) from
        GitHubSearchQuery.generator() iteration output.

        Returns a list. Each element in the list is a single dictionary representing
        an individual repository's heuristics.
    """
    # Remove headers for pagination, etc.
    ndicts = ndicts["data"]["search"]["edges"]
    # Iterate over list of nodes (repos). Length will vary
    nodes = []
    for node in ndicts:
        nodes.append(normalize(node))
    return nodes


def main():
    """ Test the class implementations """

    fetch_github = gqlquery.GitHubSearchQuery(PAT, maxitems=2)
    github_generator = fetch_github.generator()
    gdata = next(github_generator)
    print_json(gdata)
    clean_gdata = GitStarTransform(gdata)
    clean_gdata = clean_gdata.transform()
    print(type(clean_gdata))
    print(type(gdata))
    print_json(gdata)
    print_json(clean_gdata)
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
