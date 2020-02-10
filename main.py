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

def normalize(obj):
    # Remove header
    obj = obj["node"]
    # Normalize to depth=1

def transform(obj):
    # fetch repo nodes as list
    obj = obj["data"]["search"]["edges"]
    # iterate over list
    return obj


def main():
    """ Test the class implementations """
    fetch_github = gqlquery.GitHubSearchQuery(PAT, maxitems=2)
    github_generator = fetch_github.generator()
    gdata = transform(next(github_generator))
    print_json(gdata)




   #while True:
   # for _ in range(3):
   #     # Print until StopIteration generator exception
   #     print_json(next(github_generator))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
