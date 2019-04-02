from collections import defaultdict

prefixes_count = defaultdict(int)


def generate(prefix):
    name = "{}_{}".format(prefix, prefixes_count[prefix])
    prefixes_count[prefix] = prefixes_count[prefix] + 1
    return name
