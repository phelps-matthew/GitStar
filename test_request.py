""" Extract and print graphql query response from GitHub """
import json
import gqlquery

# PAT = "PERSONAL_ACCESS_TOKEN"
PAT = "423f853ce3ba7ed0bb0588e3fa8588f970a58923"


def json_str(json_dict):
    """ Use json.dumps to convert json to printable str """
    return json.dumps(json_dict, indent=4)


def print_json(json_dict):
    """ Prints nice json string """
    print(json.dumps(json_dict, indent=4))


def main():
    """ Test the class implementations """
    a2 = gqlquery.GitStarQuery(PAT, maxitems=1)
    b = a2.generator()
    jprint(next(b))
    jprint(next(b))
    jprint(next(b))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
nextpage
printjson
print_json
