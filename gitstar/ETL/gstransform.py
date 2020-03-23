"""
Cleans and transforms output from GitHubSearchQuery GraphQL response.
* Flattens all nested dicts within single node (repo) to depth=1
* Adds int timestamp date key:value pairs
* Provides method to collect repo count within GraphQL response
"""

import logging
import arrow


def transform(ndicts):
    """
    Normalizes json decoded graphql response (a nested dictionary) from
    direct output of GitHubSearchQuery generator.

    Warning: Input is transformed in place!

    Parameters
    ----------
    ndicts : dict
        Nested dictionary; output of GitHubSearchQuery.generator()

    Returns
    -------
    nodes : list
        Elements composed of dicts representing individual repo. heuristics
    """
    # Remove pagination and other headers
    ndicts = ndicts["data"]["search"]["edges"]

    # Iterate over list of nodes (repos); length will vary
    # Normalize and convert date to int for each node (repo)
    nodes = []
    for node in ndicts:
        nodes.append(add_intdate(normalize(node)))
    return nodes


def normalize(ndict):
    """
    Remove nested fields from single node (repo)

    Hard coded keylists for speed, as opposed to iterating through
    all keys. Less general, but faster.

    Warning: Input is transformed in place!

    Parameters
    ----------
    ndict : dict
        Represents single node (repository) from GQL response

    Returns
    -------
    ndict : dict
        Cleaned depth=1 dictionary representing single node
    """
    # Discard node header
    ndict = ndict["node"]

    # List of keys with nested dicts of depth=2
    dict_keys_d2 = [
        "licenseInfo",
        "readme_bytes",
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

    # Two for loops over key lists minimize comparisons on possible fields
    
    # Restructure depth=2 to depth=1
    for key in dict_keys_d2:
        # Catch and log null values that would otherwise be nested dicts
        try:
            value = ndict[key].popitem()[1]
            ndict[key] = value
        except AttributeError:
            logging.warning(
              "AttributeError. Null field found. key:{} repo:{}".format(
                  key, ndict["nameWithOwner"]
              )
            )
    # Restructure depth=3 to depth=1
    for key in dict_keys_d3:
        # Catch and log null values that would otherwise be nested dicts
        try:
            value = ndict[key[0]][key[1]].popitem()[1]
            ndict[key[0]] = value
        except TypeError:
            logging.warning(
              "TypeError. Null field found. key:{} repo:{}".format(
                  key, ndict["nameWithOwner"]
              )
            )
    # Return cleaned depth=1 dict
    return ndict


def add_intdate(ndict):
    """
    Add created_start and created_end int timestamp dates to node (dict)

    Warning: Transforms in place! (adds two key:value pairs)

    Parameters
    ----------
    ndict : dict
        depth=1 dict representing single node (repo)

    Returns
    -------
    ndict : dict
       Transformed dict with two additional key:value pairs
    """

    # Convert to UTC integer timestamp (seconds)
    ndict["createdAt_sec"] = int(arrow.get(ndict["createdAt"]).format("X"))
    ndict["updatedAt_sec"] = int(arrow.get(ndict["updatedAt"]).format("X"))
    return ndict


def repocount(ndicts):
    """
    Fetch repo count from json decoded graphql response header

    Parameters
    ----------
    ndicts : dict
        Nested dictionary; direct output of GitHubSearchQuery.generator()

    Returns
    -------
    node_num : int
        Number of repositories (nodes) from GQL response
    """
    node_num = ndicts["data"]["search"]["repositoryCount"]
    return node_num
