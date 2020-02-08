""" Extract and print graphql query response from GitHub """
import json
import gqlquery

# PAT = "PERSONAL_ACCESS_TOKEN"
PAT = "bdaf406ae91c51e9882c872571112dc3484b981a"


def json_to_str(json_obj):
    """ Use json.dumps to convert json to str """
    return json.dumps(json_obj, indent=4)


def main():
    """ Test the class implementations """
    myq = gqlquery.GitStarQuery(PAT)
    print(json_to_str(next(myq.response_json())))
    print(json_to_str(next(myq.response_json())))


# Run Main
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
