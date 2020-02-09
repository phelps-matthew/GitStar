""" Extract and print graphql query response from GitHub """
import config
import json
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
    print(PAT)
    # fmt: off
    import ipdb,os; ipdb.set_trace(context=5)  # noqa
    # fmt: on
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
