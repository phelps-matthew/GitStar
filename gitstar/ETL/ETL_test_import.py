# from gitstar import config  # no work
# import config  # no work
# from .. import config  # no work
# from gitstar import gqlquery  # no work
from gitstar.config_test import configx

from gqlquery import GraphQLQuery  # Sibling module works fine


print(configx.PAT)

print(dir(GraphQLQuery))
