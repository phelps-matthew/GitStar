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

# PAT = "PERSONAL_ACCESS_TOKEN"
PAT = config.PAT
# Log null return errors from graphql fields
logging.basicConfig(
    filename="logs/errors.log",
    filemode="w",
    level=logging.DEBUG,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)


def json_str(obj):
    """Serialize python object to json formatted str"""
    return json.dumps(obj, indent=4)


def print_json(obj):
    """Prints nice json string"""
    print(json.dumps(obj, indent=4))


def normalize(obj):
    """Take in single node and removes nested fields. Hard coded keylists for speed, as
        opposed to iterating through all keys. Less general, but faster."""
    # Remove node header
    obj = obj["node"]
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
    # possible fields
   
    # Restructure to depth=1.
    for key in dict_keys_d2:
        try:
            value = obj[key].popitem()[1]
            obj[key] = value
        # Catch null values that would otherwise be nested dicts. Log null errors.
        except AttributeError:
            logging.warning(
                "Null field found. Repository:{} key:{}".format(
                    obj["nameWithOwner"], key
                )
            )
    for key in dict_keys_d3:
        # Catch null values that would otherwise be nested dicts. Log null errors.
        try:
            value = obj[key[0]][key[1]].popitem()[1]
            obj[key[0]] = value
        except AttributeError:
            logging.warning(
                "Null field found. Repository:{} key:{}".format(
                    obj["nameWithOwner"], key
                )
            )
    return obj


def transform(obj):
    # Remove headers for pagination, etc. Returns list of nodes
    obj = obj["data"]["search"]["edges"]
    # Iterate over list of nodes
    nobj = []
    for i in range(len(obj)):
        nobj.append(normalize(obj[i]))
        print_json(nobj[i])
    return nobj


def main():
    """ Test the class implementations """

    fetch_github = gqlquery.GitHubSearchQuery(PAT, maxitems=2)
    github_generator = fetch_github.generator()
    gdata = next(github_generator)
    print_json(gdata)
    print_json(transform(gdata))
    # with open("gql_search_queries/query_out_sample", mode='w') as tfile:
    #   tfile.write(json.dumps(gdata, indent=4))
    print_json(gdata)


# while True:
# for _ in range(3):
#     # Print until StopIteration generator exception
#     print_json(next(github_generator))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
