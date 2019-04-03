from collections import defaultdict

prefixes_count = defaultdict(int)


# this module and function are used to generate names based on an incremental counter
# dependent on the prefix (staring from 0 index)
def generate(prefix):
    name = "{}_{}".format(prefix, prefixes_count[prefix])
    prefixes_count[prefix] = prefixes_count[prefix] + 1
    return name
