""" 
Sends and receives GraphQL  queries based on 'streaming' generator
approach. Includes pagination handling and GitHub API search query methods.

ToDo:
    Construct GitHubSearchQuery parent class for generalized queries
    Input QUERY file directory should be init. parameter of GitHubSearchQuery
    Consider including KeyError warning message (in addition to logging)
"""
import json
from time import sleep
import logging
from pathlib import Path
import requests
import arrow  # simple alt. to datetime

BASE_DIR = Path(__file__).resolve().parent


class GraphQLQuery:
    """
    Constructs graphQL queries via HTTP POST requests and collects response.

    Handles http timeouts.

    Parameters
    ----------
    headers : dict
        E.g. authorization header
    url : str
        API Endpoint
    query : str
        GQL query
    variables : dict, default None
        GQL variables

    Attributes
    ----------
    headers : dict
        E.g. authorization header
    url : str
        API Endpoint
    query : str
        GQL query
    variables : dict
        GQL variables
    """

    def __init__(self, headers, url, query, variables=None):
        self.headers = headers
        self.url = url
        self.query = query
        self.variables = variables or dict()

    def gql_response(self):
        """
        Sends HTTP POST query, retrieves response.

        Checks GQL response for error flags (such as timeouts) and, if true,
        sleeps for 60 secs. Logs requests/responses in root logger enabled at
        application level.

        Returns
        -------
        response : dict
            Returns requests.Reponse().json(), i.e. a deserialized json object.

        """
        # Construct response from intializations
        response = requests.post(
            url=self.url,
            headers=self.headers,
            json={"query": self.query, "variables": self.variables},
        ).json()

        # Check json response dict for errors from query, log result. Recursive
        if "errors" in response:
            logging.error(
                "Error graphql response from endpoint. Check logs.\n{}\n{}".format(
                    json.dumps(response, indent=4), self.variables
                )
            )
            sleep(60)
            return self.gql_response()
        return response


class GitHubGraphQLQuery(GraphQLQuery):
    """
    Initializes GQL query with Github API endpoint and authorization header.

    Parameters
    ----------
    PAT : str
        GitHub personal access token
    query : str
        GQL query
    variables : dict, default None
        GQL variables

    Attributes
    ----------
    headers : dict
        E.g. authorization header
    url : str
        API Endpoint
    query : str
        GQL query
    variables : dict
        GQL variables
    """

    ENDPOINT_URL = "https://api.github.com/graphql"

    def __init__(self, PAT, query, variables=None):
        super().__init__(
            headers={"Authorization": "token {}".format(PAT)},
            url=GitHubGraphQLQuery.ENDPOINT_URL,
            query=query,
            variables=variables,
        )


class GitStarSearchQuery(GitHubGraphQLQuery):
    """
    Sends full GitStar query and retrieves response with pagination.

    Uses GitStar specific query found in gitstar/ETL/GQL_QUERIES/QUERY. Through
    query variables, imposes search criteria "archived:false", "mirror:false",
    and "fork:true". Incorporates pagination handling method.

    Parameters
    ----------
    PAT : str
        GitHub personal access token
    created_start : arrow.arrow.Arrow, default arrow.get("2019-01-01")
        Repository creation search date begin
    created_end : arrow.arrow.Arrow, default arrow.get("2020-01-01")
        Repository creation search date end
    pushed_start : arrow.arrow.Arrow, default arrow.get("2020-01-01")
        Repository last pushed search date begin
    pushed_end : arrow.arrow.Arrow, default None
        Repository last pushed search date end
    maxitems : int, default 1
        Number of repository per page. <50 recommended.
    minstars : int, default 1
        Minimum star number for returned repositories
    maxstars: int, default None
        Maximum star number for returned repositories

    Attributes
    ----------
    headers : dict
        E.g. authorization header
    url : str
        API Endpoint
    query : str
        GQL query
    variables : dict
        GQL variables
    created_start : arrow.arrow.Arrow
        Use for date query slicing

    Notes
    -----
    Pagination field 'pageInfo' is specific to GitHub's 'search' connection,
    hence this subclass is made for searches only (as opposed to non-search
    queries).
    """

    # Read in GitStar queries from text file
    with open(BASE_DIR / "GQL_QUERIES/QUERY") as qfile, open(
        BASE_DIR / "GQL_QUERIES/TEST_QUERY"
    ) as tqfile:
        QUERY = qfile.read()
        TEST_QUERY = tqfile.read()

    SEARCH_QUERY = ["archived:false", "mirror:false", "fork:true"]

    def __init__(
        self,
        PAT,
        created_start=arrow.get("2019-01-01"),
        created_end=arrow.get("2020-01-01"),
        pushed_start=arrow.get("2020-01-01"),
        pushed_end=None,
        maxitems=1,
        minstars=1,
        maxstars=None,
    ):
        super().__init__(
            PAT=PAT, query=GitStarSearchQuery.QUERY, variables=None,
        )

        # Form GQL API variables to be passed with query
        self.variables["maxitems"] = maxitems
        self.variables["cursor"] = None

        # Copy class attribute search query
        searchq = GitStarSearchQuery.SEARCH_QUERY[:]

        # Form search string (GQL variable). A * indicates no upper bound.
        pushend = pushed_end.format("YYYY-MM-DD") if pushed_end else None
        searchq.extend(
            [
                "stars:{}..{}".format(minstars, maxstars or "*"),
                "created:{}..{}".format(
                    created_start.format("YYYY-MM-DD"),
                    created_end.format("YYYY-MM-DD"),
                ),
                "pushed:{}..{}".format(
                    pushed_start.format("YYYY-MM-DD"), pushend or "*",
                ),
            ]
        )
        self.variables["search_query"] = " ".join(searchq)

        # Helpful for date tracking at application level
        self.created_start = created_start

    def generator(self):
        """
        Pagination generator iterated upon query response boolean 'hasNextPage'.

        Calls gql_response() then updates 'cursor' and 'hasNextPage'.

        Yields
        ------
        gen : dict
            GraphQLQuery.gql_response()
        
        Notes
        -----
        Exceptions (e.g. StopIteration, RepoCountError) and are to be handled
        at application point. Logs debug info through root logger (initialized
        at application point).
        """

        # Initializations; index used for debugging
        nextpage = True
        index = 1

        while nextpage:
            try:
                gen = self.gql_response()

                # --- log/handle repository count ---
                if index == 1:
                    rep_count = gen["data"]["search"]["repositoryCount"]
                    logging.info("Repository Count: {}".format(rep_count))
                    print(
                        "Date:{}. Repository Count:{}".format(
                            self.created_start, rep_count
                        )
                    )

                    # If nodes exceed limit, log and raise error
                    if rep_count > 1000:
                        logging.warning(
                            "Repo Count {} exceeds limits!".format(rep_count)
                        )
                        raise RepoCountError(rep_count, "Repo Count exceeded!")

                # --- Acquire rate limits ---
                fuel = gen["data"]["rateLimit"]["remaining"]
                refuel_time = gen["data"]["rateLimit"]["resetAt"]

                # --- Update GQL cursor ---
                self.variables["cursor"] = gen["data"]["search"]["pageInfo"][
                    "endCursor"
                ]

                # --- Update GQL hasNextPage ---
                nextpage = gen["data"]["search"]["pageInfo"]["hasNextPage"]

                # --- Log/track GQL variables ---
                logging.info(
                    "vars:{}\nhasNextPage:{}".format(
                        json.dumps(self.variables, indent=4), nextpage
                    )
                )

                # --- Print and increment debug index ---
                print(index, end=" ")
                index += 1

                # --- Handle rate limiting from GQL score system ---
                # If out of points, sleep for extrapolated duration
                if fuel <= 1:
                    # returns datetime.timedelta obj
                    delta = arrow.get(refuel_time) - arrow.utcnow()
                    extra = 10  # add safety delay
                    logging.warning(
                        "Out of fuel. Will resume in {} seconds.".format(
                            delta.total_seconds() + extra
                        )
                    )
                    sleep(delta.total_seconds() + extra)
                    logging.warning("Refueled, resuming operation.")
                    yield gen
                else:
                    yield gen

            # --- Catch errors from malformed GQL queries ---
            except KeyError:
                logging.error(
                    "KeyError Occured!\n{}".format(json.dumps(gen, indent=4))
                )
                # Raises StopIteration to not halt application
                raise StopIteration


class RepoCountError(Exception):
    """
    If repositoryCount > 1000, raise this exception.

    Parameters
    ----------
    repo_count : int
    msg : str

    Attributes
    ----------
    repo_count : int
    msg : str
    """
    def __init__(self, repo_count, msg):
        self.repo_count = repo_count
        self.msg = msg
        super().__init__(self)
