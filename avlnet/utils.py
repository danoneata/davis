def read_file(path, parse_func=lambda x: x):
    with open(path, "r") as f:
        lines = f.readlines()
    return [parse_func(line.strip()) for line in lines]
