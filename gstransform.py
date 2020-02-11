""" GitStar transformation module. """

import json
import logging


# Log null return errors from graphql fields
logging.basicConfig(
    filename="logs/errors.log",
    filemode="w",
    level=logging.DEBUG,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)


class GitStarTransform:
    def __init__(self, decoded_json):
        self.decoded_json = decoded_json

    def transform(self):
        """Normalizes json decoded graphql data (a nested dictionary) from
            GitHubSearchQuery.generator() iteration output. Input is mutable
            object!

            Returns a list. Each element in the list is a single dictionary
            representing an individual repository's heuristics.
        """

        # ------------- Begin normalize sub-function -------- #
        def normalize(ndict):
            """Take in single node (nested dictionary representing a single repo)
                and remove nested fields. Hard coded keylists for speed, as
                opposed to iterating through all keys. Less general, but
                faster.

                Returns single dictionary.
            """
            # Remove node header
            ndict = ndict["node"]
            # List of keys with nested dicts of depth=2
            dict_keys_d2 = [
                "licenseInfo",
                "readme",
                "primaryLanguage",
                "releases",
                "stargazers",
                "watchers",
                "deployments",
                "repositoryTopics",
                "pullRequests",
                "projects",
                "milestones",
                "issuelabels",
                "closedissues",
                "openissues",
            ]
            # List of keys with nested dicts of depth=3
            dict_keys_d3 = [("commitnum", "history")]
            #
            # Two for loops are ugly, but again.. speed! We don't want
            # comparisons on all possible fields. Also faster to catch isolated
            # exception
            #
            # Restructure to depth=1.
            for key in dict_keys_d2:
                # Catch null values that would otherwise be nested dicts. Log
                # null errors.
                try:
                    value = ndict[key].popitem()[1]
                    ndict[key] = value
                except AttributeError:
                    logging.warning(
                        "Null field found. Repository:{} key:{}".format(
                            ndict["nameWithOwner"], key
                        )
                    )
            for key in dict_keys_d3:
                # Catch null values that would otherwise be nested dicts. Log
                # null errors.
                try:
                    value = ndict[key[0]][key[1]].popitem()[1]
                    ndict[key[0]] = value
                except AttributeError:
                    logging.warning(
                        "Null field found. Repository:{} key:{}".format(
                            ndict["nameWithOwner"], key
                        )
                    )
            return ndict
        # ------------- End of normalize -------------------- #

        # Remove headers for pagination, etc.
        ndicts = self.decoded_json["data"]["search"]["edges"]
        # Iterate over list of nodes (repos). Length will vary
        nodes = []
        for node in ndicts:
            nodes.append(normalize(node))
        return nodes
