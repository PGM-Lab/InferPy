from collections import defaultdict

# for each used prefix, return a different counter starting from 0
prefixes_count = defaultdict(int)


def generate(prefix):
    """This function is used to generate names based on an incremental counter (global variable in this module)
        dependent on the prefix (staring from 0 index)

        :prefix (`str`): The begining of the random generated name

        :returns: The generated random name
    """
    name = "{}_{}".format(prefix, prefixes_count[prefix])
    prefixes_count[prefix] = prefixes_count[prefix] + 1
    return name
