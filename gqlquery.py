""" Implementation of graphQL HTTP POST queries based on 'streaming' generator
    approach. Includes pagination handling and GitHub API specific methods.
"""
import requests


class GraphQLQuery:
    """ Send json graphql query request and query variables dict via HTTP POST
    requests
    """

    def __init__(self, headers, url, query, variables=None):
        self.headers = headers
        self.url = url
        self.query = query
        # returns empty dict() if variables=None
        self.variables = variables or dict()

    def gql_response(self):
        """ Sends HTTP POST query. Returns generator containing a
            requests.Reponse().json() object """
        try:
            return requests.post(
                url=self.url,
                headers=self.headers,
                json={"query": self.query, "variables": self.variables},
            ).json()

        # ! Need to look into these exceptions more

        except requests.exceptions.HTTPError as http_err:
            raise http_err
        except Exception as err:
            raise err



class GitHubGraphQLQuery(GraphQLQuery):
    """ Incorporates github header and endpoint """

    ENDPOINT_URL = "https://api.github.com/graphql"

    def __init__(self, PAT, query, variables):
        super().__init__(
            headers={"Authorization": "token {}".format(PAT)},
            url=GitHubGraphQLQuery.ENDPOINT_URL,
            query=query,
            variables=variables,
        )


class GitStarQuery(GitHubGraphQLQuery):
    """ Implements graphql query to fetch filtered repos
        and fields based on GitStar criterium.

        Input: PAT.
        Methods: .next, .json, .text, .status_code

        generators have to be methods

        Each instance should be a different generator.

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
       search(query: $search_query, type: REPOSITORY, first: $maxitems, """\
       """after: $cursor)
       {
         pageInfo {
           endCursor
           hasNextPage
         }
         repositoryCount
         edges {
           node {
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

    VARIABLES = {
        "search_query": "archived:false mirror:false stars:>0 "\
        "created:>=2015-01-01 pushed:>=2019-01-01 fork:true",
        "maxitems": 1,
        "after": None,
    }

    def __init__(self, PAT, maxitems=1):
        super().__init__(
            PAT=PAT, query=GitStarQuery.QUERY, variables=GitStarQuery.VARIABLES
        )
        self.variables["maxitems"] = maxitems

    def my_gen(self):
        """ Pagination generator iterated upon query response boolean hasNextPage.
            Calls gql_response() then updates cursor and hasNextPage.
            Exceptions (e.g. StopIteration) are to be handled outside of class
        """
        hasnextpage = True
        while hasnextpage:
            resp = self.gql_response()
            # Update cursor
            self.variables["after"] =\
                resp["data"]["search"]["pageInfo"]["endCursor"]
            # Update hasNextPage
            hasnextpage = resp["data"]["search"]["pageInfo"]["hasNextPage"]
            yield resp
