""" Extract and print graphql query response from GitHub """
import json
import gqlquery

# PAT = "PERSONAL_ACCESS_TOKEN"


def jstr(json_obj):
    """ Use json.dumps to convert json to nice str """
    return json.dumps(json_obj, indent=4)


def jprint(json_obj):
    """ Prints nice json string """
    print(json.dumps(json_obj, indent=4))


def main():
    """ Test the class implementations """
    a2 = gqlquery.GitStarQuery(PAT, maxitems=10)
    b = a2.generator()
    jprint(next(b))
    jprint(next(b))
    jprint(next(b))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
hasNextPage
