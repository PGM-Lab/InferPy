import pytest
from tensorflow_probability.python import edward2 as ed

from inferpy.util import tf_graph


# elements in the same second lists are dependent
@pytest.mark.parametrize("dependencies1, dependencies2", [
    # two independent variables
    (['x'], ['y']),
    # two sets of independent variables, and each group of three dependent variables (chain)
    (['x', 'y', 'z'], ['u', 'v', 'w'])
])
def test_dependencies(dependencies1, dependencies2):
    # register and check dependencies between group of dependent vars
    for dependent_vars in (dependencies1, dependencies2):
        last = ed.Normal(0, 1, name=dependent_vars[0])
        for i in range(1, len(dependent_vars)):
            parentname = dependent_vars[i - 1]
            name = dependent_vars[i]
            x = ed.Normal(last, 1, name=name)
            g = tf_graph.get_graph(set([parentname, name]))
            assert parentname in g.predecessors(name)
            last = x
    # all variables registered. Now check independencies between independent groups
    g = tf_graph.get_graph(set(dependencies1 + dependencies2))
    for v1 in dependencies1:
        for v2 in dependencies2:
            assert v1 not in g.predecessors(v2)
            assert v2 not in g.predecessors(v1)
