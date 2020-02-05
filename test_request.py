""" Module doctring! """
import json
import requests

# Globals
ENDPOINT_URL = "https://api.github.com/graphql"
PAT = "a20a63ae728bf49e6cb5f097a6832de4f70595b9"

header = {"Authorization": "token " + PAT}

myquery = """\
{
    viewer {
        name
    }
    rateLimit {
        limit
        cost
        remaining
        resetAt
    }
    rep2: repository(name: "nbconvert", owner: "jupyter") {
        nameWithOwner
    }
    rep1: repository(name: "ace", owner: "ajaxorg") {
        nameWithOwner
    }
}
"""


def print_json(obj):
    """ prints nice json using json.dumps to convert json to str """
    text = json.dumps(obj, indent=4)
    print(text)


# Returns a requests.Reponse() object
r = requests.post(url=ENDPOINT_URL, json={"query": myquery}, headers=header)
print(r.status_code)
print_json(r.json())
