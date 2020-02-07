import graphQLquery

# PAT = "PERSONAL_ACCESS_TOKEN"
PAT = "d3ed40f3499144ebf755b813e20bdc05b2ede08b"


def main():
    """ Test the class implementations """
    myq = graphQLquery.GitStarQuery(PAT)
    print(r.status_code)
    print_json(r.json())


# Run Main
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
