""" Implementation of graphQL HTTP POST queries based on 'streaming' generator
    approach. Includes pagination handling and GitHub API specific methods.
"""
import json
import requests


def json_str(json_dict):
    """ Use json.dumps to convert json to printable str """
    return json.dumps(json_dict, indent=4)


def print_json(json_dict):
    """ Prints nice json string """
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
        """Sends HTTP POST query. Returns generator containing a
            requests.Reponse().json() object.
        """
        req = requests.post(
            url=self.url,
            headers=self.headers,
            json={"query": self.query, "variables": self.variables},
        ).json()
        if "errors" in req:
            print_json(req)
            raise requests.RequestException(
                "Error graphql response from endpoint. Check query"
            )
        return req


class GitHubGraphQLQuery(GraphQLQuery):
    """Base class for any GitHub graphQL query. Incorporates github header and
        endpoint
    """

    ENDPOINT_URL = "https://api.github.com/graphql"

    def __init__(self, PAT, query, variables):
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

    QUERY = """\
    query searchmp($search_query: String!, $maxitems: Int, $cursor: String) {
       rateLimit {
         limit
         cost
         remaining
         resetAt
       }
       search(query: $search_query, type: REPOSITORY, """\
        """first: $maxitems, after: $cursor)
       {
         pageInfo {
           endCursor
           hasNextPage
         }
         repositoryCount
         edges {
           node {
              # Only __typename is Repository
              ... on Repository {
               nameWithOwner
               readme: object(expression: "master:README.md") {
                 ... on Blob {
                   byteSize
                 }
               }
               shortDescriptionHTML
               id
               databaseId
               createdAt
               updatedAt
               forkCount
               diskUsage
               releases {
                 totalCount
               }
               stargazers {
                 totalCount
               }
               watchers {
                 totalCount
               }
               deployments {
                 totalCount
               }
             }
           }
         }
        }
      }
     """

    TEST_QUERY = """\
    query searchmp($search_query: String!, $maxitems: Int, $cursor: String) {
        rateLimit {
          limit
          cost
          remaining
          resetAt
        }
        search(query: $search_query, type: REPOSITORY, \
        first: $maxitems, after: $cursor) {
          pageInfo {
            endCursor
            hasNextPage
          }
          repositoryCount
          edges {
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
        "created:>=2020-02-01 pushed:>=2020-01-01 fork:true",
        "maxitems": 1,
        "cursor": None,
    }

    def __init__(self, PAT, maxitems=1):
        super().__init__(
            PAT=PAT,
            query=GitStarQuery.TEST_QUERY,
            variables=GitStarQuery.VARIABLES,
        )
        self.variables["maxitems"] = maxitems

    def generator(self):
        """Pagination generator iterated upon query response boolean 'hasNextPage'.
            Calls gql_response() then updates cursor and hasNextPage.
            Exceptions (e.g. StopIteration) are to be handled outside of class.
        """
        nextpage = True
        while nextpage:
            gen = self.gql_response()
            # Update cursor
            self.variables["cursor"] = \
                gen["data"]["search"]["pageInfo"]["endCursor"]
            # Update hasNextPage
            nextpage = gen["data"]["search"]["pageInfo"]["hasNextPage"]
            yield gen
