import config
import json
import arrow
from time import sleep
import requests

PAT = config.PAT
ENDPOINT_URL = "https://api.github.com/graphql"
headers = {"Authorization": "token {}".format(PAT)}

TEST_QUERY = """\
    query searchmp($search_query: String!, $maxitems: Int, $cursor: String) {
        rateLimit {
          limit
          cost
          remaining
          resetAt
        }
        search(query: $search_query, type: REPOSITORY, first: $maxitems, after: $cursor) {
          pageInfo {
            endCursor
            hasNextPage
          }
          repositoryCount
          edges {
            cursor
            node {
              # Only __typename is Repository
              ... on Repository {
               nameWithOwner
              }
            }
          }
        }
    }
    """

VARIABLES = {
    "search_query": "archived:false mirror:false stars:>0 "
    "created:>=2020-02-01 pushed:>=2019-01-01 fork:true",
    "maxitems": 2,
    "after": None,
}


def json_str(json_obj):
    """ Use json.dumps to convert json to nice str """
    return json.dumps(json_obj, indent=4)


def print_json(json_obj):
    """ Prints nice json string """
    print(json.dumps(json_obj, indent=4))


def fetch_req():
    req = requests.post(
        url=ENDPOINT_URL,
        headers=headers,
        json={"query": TEST_QUERY, "variables": VARIABLES},
    ).json()
    if "errors" in req:
        print_json(req)
        raise requests.RequestException(
            "Error graphql response from endpoint. Check query"
        )
    return req


def main():
    gen = fetch_req()
    fuel = gen["data"]["rateLimit"]["remaining"]
    refuel_time = gen["data"]["rateLimit"]["resetAt"]
    delta = arrow.get(refuel_time) - arrow.utcnow()
    #sleep(delta.seconds)
    print("fuel:{}\nrefuel_time:{}".format(fuel, refuel_time))
    print(arrow.utcnow())
    print(arrow.get(refuel_time))
    print(delta)
    print(delta.seconds)
    print_json(gen)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()

# iterator = iter(iterable)
# try:
#    while True:
#        item = next(iterator)
#        do_stuff(item)
# except StopIteration:
#    pass
# finally:
#    del iterator

# while True:
#    try:
#        obj = next(my_gen)
#    except StopIteration:
#        break
#
# print('Done')
