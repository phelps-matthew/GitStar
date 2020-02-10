""" Extract and print graphql query response from GitHub """
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
    gs = gqlquery.GitStarQuery(PAT, maxitems=1)
    gs_gen = gs.generator()

    while True:
        # Print until StopIteration generator exception
        print_json(next(gs_gen))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
