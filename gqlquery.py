""" Implementation of graphQL HTTP POST queries based on 'streaming' generator
    approach. GitHub API classes available.  """
import requests


class GraphQLQuery:
    """ Send json graphql query request and
        query variables dict via HTTP POST requests """

    def __init__(self, headers, url, query, variables=None):
        self.headers = headers
        self.url = url
        self.query = query
        # returns empty dict() if variables=None
        self.variables = variables or dict()

    def response_json(self):
        """ Returns a requests.Reponse() object.
            Acts as generator to be iterated upon. """
        while True:
            try:
                yield requests.post(
                    url=self.url,
                    headers=self.headers,
                    json={"query": self.query, "variables": self.variables},
                ).json()
            except requests.exceptions.HTTPError as http_err:
                raise http_err
            except Exception as err:
                raise err


class GitHubGraphQLQuery(GraphQLQuery):
    """ has github header, endpoint """

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

        cursors, nextpages is query specific. put here """

    QUERY = """\
    query searchmp($myq: String!, $maxItems: Int, $cursor: String) {
       rateLimit {
         limit
         cost
         remaining
         resetAt
       }
       search(query: $myq, type: REPOSITORY, first: $maxItems, after: $cursor) {
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
        "myq": "archived:false mirror:false stars:>0\
         created:>=2015-01-01 pushed:>=2019-01-01 fork:true",
        "maxItems": 5,
    }

    def __init__(self, PAT, hasNextPage=True):
        super().__init__(
            PAT=PAT, query=GitStarQuery.QUERY, variables=GitStarQuery.VARIABLES
        )
        self.hasNextPage = hasNextPage
