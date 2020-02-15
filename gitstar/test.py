def out():
    x = 9
    def main():
        for _ in range(3):
            nonlocal x
            x = 9
            try:
                x = x + 1
                print(x)
            except StopIteration:
                x = x + 3
                print(x)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
