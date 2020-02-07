""" module docstring """
import json
import requests

# PAT = "PERSONAL_ACCESS_TOKEN"


class GraphQLQuery:
    """ Send json graphql queries via HTTP POST requests. This class is probably redundant\
    as it only simplifies use of query and variables as solely compared to requests """

    def __init__(self, headers, url, query, variables=None):
        self.headers = headers
        self.url = url
        self.query = query
        # returns empty dict() if variables=None
        self.variables = variables or dict()


    def request():
        # Returns a requests.Reponse() object
        r = requests.post(
            self.url, 
            json={"query": self.query, "variables": self.variables},
            headers=self.headers)


class GitHubGraphQLQuery(GraphQLQuery):
    """ has github header, endpoint """

    ENDPOINT_URL = "https://api.github.com/graphql"

    def __init__(self, PAT, query, variables):
        super().__init__(headers={"Authorization": "token {}".format(PAT)},\
                         url=ENDPOINT_URL,
                         query=query, 
                         variables=variables)


class GitStarQuery(GitHubGraphQLQuery):
    """ Implements graphql query to fetch filtered repos\
        and fields based on GitStar criterium """

    QUERY = """\
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

    def __init__(self, PAT):
        super().__init__(PAT=PAT, query=QUERY, variables=None)

def print_json(obj):
    """ prints nice json using json.dumps to convert json to str """
    text = json.dumps(obj, indent=4)
    print(text)



print(r.status_code)
print_json(r.json())
