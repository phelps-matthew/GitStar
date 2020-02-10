""" ETL main application entry point. Extract github data. Transform and clean relevant
fields. Load into SQL database
"""
import json
import config
import gqlquery

# PAT = "PERSONAL_ACCESS_TOKEN"
PAT = config.PAT


def json_str(obj):
    """Serialize python object to json formatted str"""
    return json.dumps(obj, indent=4)


def print_json(obj):
    """Prints nice json string"""
    print(json.dumps(obj, indent=4))


def main():
    """ Test the class implementations """
    fetch_github = gqlquery.GitHubSearchQuery(PAT, maxitems=1)
    github_generator = fetch_github.generator()
    # fmt: off
    import ipdb,os; ipdb.set_trace(context=5)  # noqa
    # fmt: on
     

    #while True:
    for _ in range(3):
        # Print until StopIteration generator exception
        print_json(next(github_generator))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
