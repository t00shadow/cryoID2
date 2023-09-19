dbsize = 0

with open("test.txt", "r") as f2:
    for line in f2:
        if not line.startswith(">"):
            line = line.strip()
            dbsize += len(line)

print(dbsize)
