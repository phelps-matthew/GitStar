""" Implementation of graphQL HTTP POST queries based on 'streaming' generator
    approach. Includes pagination handling and GitHub API specific methods.
"""
import json
from time import sleep
import requests
import arrow


def json_str(json_dict):
    """Use json.dumps to convert json to printable str"""
    return json.dumps(json_dict, indent=4)


def print_json(json_dict):
    """Prints nice json string"""
    print(json.dumps(json_dict, indent=4))


class GraphQLQuery:
    """Base class for graphQL queries. Sends json graphql query request and
        query variables dict via HTTP POST requests.
    """

    def __init__(self, headers, url, query, variables=None):
        self.headers = headers
        self.url = url
        self.query = query
        # returns empty dict() if variables=None
        self.variables = variables or dict()

    def gql_response(self):
        """Sends HTTP POST query. Returns requests.Reponse().json() object."""
        response = requests.post(
            url=self.url,
            headers=self.headers,
            json={"query": self.query, "variables": self.variables},
        ).json()

        # Check json response dict for errors from query
        if "errors" in response:
            print_json(response)
            raise requests.RequestException(
                "Error graphql response from endpoint. Check query!"
            )
        return response


class GitHubGraphQLQuery(GraphQLQuery):
    """Base class specific to GitHub graphQL queries. Incorporates github
        authorization header and endpoint
    """

    ENDPOINT_URL = "https://api.github.com/graphql"

    def __init__(self, PAT, query, variables=None):
        super().__init__(
            headers={"Authorization": "token {}".format(PAT)},
            url=GitHubGraphQLQuery.ENDPOINT_URL,
            query=query,
            variables=variables,
        )


class GitStarQuery(GitHubGraphQLQuery):
    """Implements graphql query to fetch filtered repos
        and fields based on GitStar criterium.

        cursors, nextpages is query specific. put here
    """

    # Read in queries from text file
    with open("gql_queries/QUERY") as qfile,\
         open("gql_queries/TEST_QUERY") as tqfile:
        QUERY = qfile.read()
        TEST_QUERY = tqfile.read()

    VARIABLES = {
        "search_query": "archived:false mirror:false stars:>100 "
                        "created:>=2020-02-01 pushed:>=2020-01-01 fork:true",
        "maxitems": 1,
        "cursor": None,
    }

    def __init__(self, PAT, maxitems=1):
        super().__init__(
            PAT=PAT, query=GitStarQuery.TEST_QUERY, variables=GitStarQuery.VARIABLES,
        )
        # Add configurable maxitems instance attribute
        self.variables["maxitems"] = maxitems

    def generator(self):
        """Pagination generator iterated upon query response boolean 'hasNextPage'.
            Calls gql_response() then updates cursor and hasNextPage.
            Exceptions (e.g. StopIteration) are to be handled outside of class.
        """
        nextpage = True
        while nextpage:
            gen = self.gql_response()
            # Acquire rate limits
            fuel = gen["data"]["rateLimit"]["remaining"]
            refuel_time = gen["data"]["rateLimit"]["resetAt"]
            # Update cursor
            self.variables["cursor"] = gen["data"]["search"]["pageInfo"]["endCursor"]
            # Update hasNextPage
            nextpage = gen["data"]["search"]["pageInfo"]["hasNextPage"]
            # Handle rate limiting
            if fuel <= 1:
                # returns datetime.timedelta obj
                delta = arrow.get(refuel_time) - arrow.utcnow()
                extra = 10  # additional delay
                print(
                    "[{}] Out of fuel. Will resume in {} seconds.".format(
                        arrow.now(), delta.seconds + extra
                    )
                )
                sleep(delta.seconds + extra)
                print("[{}] Refueled, resuming operation.".format(arrow.now()))
                yield gen
            else:
                yield gen
