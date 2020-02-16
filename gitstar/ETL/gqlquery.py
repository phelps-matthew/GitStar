""" Implementation of graphQL HTTP POST queries based on 'streaming' generator
    approach. Includes pagination handling and GitHub API specific methods.

    TODO:
        Input QUERY file directory should be class init. parameter

        Put generator in parent class, initialized with query specific
        connection parameters

        Improve docstrings
"""
import json
from time import sleep
import requests
import arrow  # simple alt. to datetime
import logging


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
        """Sends HTTP POST query. Returns requests.Reponse().json() object.
            The return is a deserialized json object, i.e. a python dict
        """
        response = requests.post(
            url=self.url,
            headers=self.headers,
            json={"query": self.query, "variables": self.variables},
        ).json()

        # Check json response dict for errors from query. Recursive.
        if "errors" in response:
            logging.error(
                "Error graphql response from endpoint. Check logs.\n{}".format(
                    json.dumps(response, indent=4)
                )
            )
            sleep(60)
            return self.gql_response()
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


class GitHubSearchQuery(GitHubGraphQLQuery):
    """Implements graphql search query to fetch fields from GitHub. Pagination
        field 'pageInfo' is specific to GitHub's 'search' connection, hence
        this subclass is made for searches only.
    """

    # Read in custom queries from text file
    with open("GQL_QUERIES/QUERY") as qfile, open(
        "GQL_QUERIES/TEST_QUERY"
    ) as tqfile:
        QUERY = qfile.read()
        TEST_QUERY = tqfile.read()

    SEARCH_QUERY = ["archived:false", "mirror:false", "fork:true"]

    def __init__(
        self,
        PAT,
        created_start=arrow.get("2021-01-01"),
        created_end=arrow.get("2021-02-01"),
        last_pushed=arrow.get("2020-01-01"),
        maxitems=1,
        stars=0,
    ):
        super().__init__(
            PAT=PAT, query=GitHubSearchQuery.QUERY, variables=None,
        )
        # Form gql API variables to be passed with query
        self.variables["maxitems"] = maxitems
        self.variables["cursor"] = None
        # Copy class attribute list
        searchq = GitHubSearchQuery.SEARCH_QUERY[:]
        searchq.extend(
            [
                "stars:>{}".format(stars),
                "created:{}..{}".format(
                    created_start.format("YYYY-MM-DD"),
                    created_end.format("YYYY-MM-DD"),
                ),
                "pushed:{}..*".format(last_pushed.format("YYYY-MM-DD")),
            ]
        )
        self.variables["search_query"] = " ".join(searchq)

    def generator(self):
        """Pagination generator iterated upon query response boolean 'hasNextPage'.
            Calls gql_response() then updates cursor and hasNextPage.
            Exceptions (e.g. StopIteration) are to be handled outside of class.
            Generator is composed of nested dicts, deserialized from json
            response.
        """
        nextpage = True
        index = 1  # For debugging
        while nextpage:
            gen = self.gql_response()
            # log repository count
            #if index == 1:
               # logging.info(
               #     "Repository Count: {}".format(
               #         gen["data"]["search"]["repositoryCount"]
               #     )
               # )
               # print(
               #     "Repository Count: {}".format(
               #         gen["data"]["search"]["repositoryCount"]
               #     )
               # )
            # Acquire rate limits
            fuel = gen["data"]["rateLimit"]["remaining"]
            refuel_time = gen["data"]["rateLimit"]["resetAt"]
            # Update cursor
            self.variables["cursor"] = gen["data"]["search"]["pageInfo"][
                "endCursor"
            ]
            # Update hasNextPage
            nextpage = gen["data"]["search"]["pageInfo"]["hasNextPage"]
            #logging.info(
            #    "Cursor: {} hasNextPage:{}".format(
            #        self.variables["cursor"], nextpage
            #    )
            # )
            #logging.info(json.dumps(self.variables, indent=4))
            #print(index)
            index += 1
            # Handle rate limiting
            if fuel <= 1:
                # returns datetime.timedelta obj
                delta = arrow.get(refuel_time) - arrow.utcnow()
                extra = 10  # additional delay
                logging.info(
                    "Out of fuel. Will resume in {} seconds.".format(
                        delta.total_seconds() + extra
                    )
                )
                sleep(delta.total_seconds() + extra)
                logging.info("Refueled, resuming operation.")
                yield gen
            else:
                yield gen
