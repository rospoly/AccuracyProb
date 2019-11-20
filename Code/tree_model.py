from error_model import ErrorModel


def copy_tree(my_tree):
    if my_tree.leaf:
        copied_tree = BinaryTree(my_tree.value.name, my_tree.value)
    else:
        copied_tree = BinaryTree(my_tree.value.operator, None, copy_tree(my_tree.children[0]), copy_tree(my_tree.children[1]))
    return copied_tree


class BinaryTree(object):
    def __init__(self, name, value, left=None, right=None):
        self.root_name = name
        self.root_value = value
        self.left = left
        self.right = right


class TreeModel:
    def __init__(self, my_yacc, precision, exp, poly_precision, initialize=False):
        self.initialize = initialize
        self.precision = precision
        self.exp = exp
        self.poly_precision = poly_precision
        ''' Copy structure of the tree from my_yacc'''
        self.tree = copy_tree(my_yacc.expression)
        ''' Evaluate tree '''
        self.evaluate(self.tree)

    # Recursively populate the Tree with the triples (distribution, error distribution, quantized distribution)
    def evaluate(self, tree):
        triple = []
        # Test if we're at a leaf
        if tree.root_value is not None:
            # Non-quantized distribution
            dist = tree.root_value.distribution
            # initialize=True means we quantize the inputs
            if self.initialize:
                # Compute error model
                error = ErrorModel(dist, self.precision, self.exp, self.poly_precision)
                quantized_distribution = dist*(1+error)
            # Else we leave the leaf distribution unchanged
            else:
                error = 0
                quantized_distribution = dist
        # If not at a leaf we need to get the distribution and quantized distributions of the children nodes.
        # Then, check the operation. For each operation the template is the same:
        # dist will be the non-quantized operation the non-quantized children nodes
        # qdist will be the non-quantized operation on the quantized children nodes
        # quantized_distribution will be the quantized operation on the quantized children nodes
        else:
            self.evaluate(tree.left)
            self.evaluate(tree.right)
            dist = BinOpDist(tree.left.root_value[0], tree.root_name, tree.right.root_value[0])
            qdist = BinOpDist(tree.left.root_value[2], tree.root_name, tree.right.root_value[2])
            error = ErrorModel(qdist, self.precision, self.exp, self.poly_precision)
            quantized_distribution = BinOpDist(qdist.execute(), "*", (1 + error.distribution))
        # We now populate the triple with distribution, error model, quantized distribution
        triple.append(dist)
        triple.append(error)
        triple.append(quantized_distribution)
        tree.root_value = triple


class BinOpDist:
    def __init__(self, leftoperand, operator, rightoperand):
        self.leftoperand = leftoperand
        self.operator = operator
        self.rightoperand = rightoperand

        if operator=="+":
            self.distribution = self.leftoperand + self.rightoperand
        elif operator=="-":
            self.distribution = self.leftoperand - self.rightoperand
        elif operator=="*":
            self.distribution = self.leftoperand * self.rightoperand
        elif operator=="/":
            self.distribution = self.leftoperand / self.rightoperand
        else:
            print ("Operation not supported!")
            exit(-1)
        ####################################
        self.a = self.distribution.a
        self.b = self.distribution.b

    def execute(self):
        return self.distribution