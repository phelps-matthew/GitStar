""" ETL main application entry point. Extract github data. Transform and clean relevant
fields. Load into SQL database
"""
import json
import config
import gqlquery

# PAT = "PERSONAL_ACCESS_TOKEN"
PAT = config.PAT


def json_str(json_dict):
    """ Use json.dumps to convert json to printable str """
    return json.dumps(json_dict, indent=4)


def print_json(json_dict):
    """ Prints nice json string """
    print(json.dumps(json_dict, indent=4))


def main():
    """ Test the class implementations """
    fetch_github = gqlquery.GitStarQuery(PAT, maxitems=1)
    fetch_github_gen = fetch_github.generator()

    #while True:
    for _ in range(3):
        # Print until StopIteration generator exception
        print_json(next(fetch_github_gen))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
