""" module docstring """
import json
import requests


class GraphQLQuery:
    """ Send json graphql queries via HTTP POST requests.\
        This class is probably redundant as it only simplifies\
        use of query and variables as solely compared to requests """

    def __init__(self, headers, url, query, variables=None):
        self.headers = headers
        self.url = url
        self.query = query
        # returns empty dict() if variables=None
        self.variables = variables or dict()

    def request(self):
        """ Returns a requests.Reponse() object """
        return requests.post(
            url=self.url,
            json={"query": self.query, "variables": self.variables},
            headers=self.headers,
        )

    def text(self):
        """ printable text output using json.dumps to convert json to str """
        return request(self).json().dumps(self, indent=4)

    def status_code(self):
        """ printable text output using json.dumps to convert json to str """
        return request(self).status_code


class GitHubGraphQLQuery(GraphQLQuery):
    """ has github header, endpoint """

    ENDPOINT_URL = "https://api.github.com/graphql"

    def __init__(self, PAT, query, variables):
        super().__init__(
            headers={"Authorization": "token {}".format(PAT)},
            url=ENDPOINT_URL,
            query=query,
            variables=variables,
        )


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

    VARIABLES = {tbd}

    def __init__(self, PAT):
        super().__init__(PAT=PAT, query=QUERY, variables=VARIABLES)
