import graphQLquery

# PAT = "PERSONAL_ACCESS_TOKEN"


def main():
    """ Test the class implementations """
    myq = graphQLquery.GitStarQuery(PAT)
    myq.json
    print(myq.text)
    print(myq.status_code)
    print(r.status_code)
    print_json(r.json())


# Run Main
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
