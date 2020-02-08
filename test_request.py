""" Extract and print graphql query response from GitHub """
import gqlquery

PAT = "PERSONAL_ACCESS_TOKEN"


def main():
    """ Test the class implementations """
    myq = gqlquery.GitStarQuery(PAT)
    print(next(myq.response_json()))


# Run Main
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
