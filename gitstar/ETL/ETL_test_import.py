# from gitstar import config  # no work
# import config  # no work
# from .. import config  # no work
# from gitstar import gqlquery  # no work
from gitstar.configtest import configx  # Sibling package does not work

from gqlquery import GraphQLQuery  # Sibling module works fine


print(configx.PAT)

print(dir(GraphQLQuery))
