from collections import defaultdict
import networkx as nx
import tensorflow as tf


"""
These set of private functions obtain the dependencies between Random Variables
by analyzing the tensors in Random Varibles and its relations. Finally, the
_get_graph function computes a graph in networkx that represents these dependencies.
"""


def _get_varname(op):
    op_name = op.name
    idx = op_name.find('/')  # Use the first part of the operation name (until slash) as name
    if idx != -1 and '/Assign' not in op_name:  # Special case for tf.Variables
        return op_name[:idx]
    else:
        return op_name


def _children(op):
    # get the consumers of the operation as its children (set of names, using _get_varname)
    return set(_get_varname(opc) for out in op.outputs for opc in out.consumers())


def _clean_graph(G, varnames):
    # G is a networkx graph. Clean nodes from it whose names are not in varnames set or dict.
    # Before removing such nodes, create an edge between their parents and their children.
    g_nodes = list(G.nodes)
    for n in g_nodes:
        if n not in varnames:
            if '/Assign' in n:  # Special case for tf.Variables
                predecesors = list(G.predecessors(n))
                n_name = n[n.rfind('/')]  # real name of the tf.variable

                assert len(predecesors) <= 2  # At most, it should have two predecessors

                if len(predecesors) == 2:
                    if predecesors[0] == n_name:
                        # create relation from predecesors[1] to predecesors[0]
                        G.add_edge(predecesors[1], predecesors[0])
                    else:
                        # create relation from predecesors[0] to predecesors[1]
                        G.add_edge(predecesors[0], predecesors[1])
            else:
                # remove and create edge between parent and child if exist
                for p in G.predecessors(n):
                    for s in G.successors(n):
                        G.add_edge(p, s)
            G.remove_node(n)
    return G


def get_graph(varnames):
    # varnames is a set or dict where keys are the var names of the Random Variables

    # TODO: using default graph to build the model and get the graph.
    # In the future we should use a new graph, or at least allow to give a
    # new one as parameter.

    # Creates dictionary {node: {child1, child2, ..},..} for current
    # TensorFlow graph. Result is compatible with networkx/toposort
    ops = tf.get_default_graph().get_operations()
    dependencies = defaultdict(set)
    for op in ops:
        # in tensorflow_probability, the tensor named *sample_shape* is a op in child-parent order.
        # as we want to capture only the parent-child relations, skip these op names
        # TODO: This is not robust, because sample shape tensors might use a different name (this is the default name)
        if 'sample_shape' not in op.name:
            c = _children(op)
            # if len(c) > 0:
            op_name = _get_varname(op)
            c.discard(op_name)  # avoid name references to itself
            dependencies[op_name].update(c)

    # create networkx graph
    G = nx.DiGraph(dependencies)
    # clean names, to get just ed2 RV var names
    _clean_graph(G, varnames)  # inplace modification
    return G


def get_empty_graph():
    return nx.DiGraph()
