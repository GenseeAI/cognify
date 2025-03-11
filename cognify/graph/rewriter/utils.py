import ast
import astunparse
from typing import Callable
import inspect


# NOTE: this only works with functions/methods but not lambda expression
def add_argument_to_position(signature: str, new_arg, i=0) -> tuple[str, str]:
    """return [new function, function name]"""
    tree = ast.parse(signature)

    # Get the function definition node
    func_def = tree.body[0]
    new_arg_node = ast.arg(arg=new_arg, annotation=None)
    func_def.args.args.insert(i, new_arg_node)

    # Convert the AST back to a string
    new_signature = astunparse.unparse(tree).strip()
    return new_signature, func_def.name


class RewriteBranch(ast.NodeTransformer):
    def __init__(self, old_name, new_name):
        super().__init__()
        self.old_name = old_name
        self.new_name = new_name

    def visit_Return(self, node):
        if isinstance(node.value, ast.Constant) and node.value.value == self.old_name:
            node.value.value = self.new_name
        if isinstance(node.value, ast.List):
            for elem in node.value.elts:
                if isinstance(elem, ast.Constant) and elem.value == self.old_name:
                    elem.value = self.new_name
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "hint_possible_destinations"
            ):
                # change constant in args to new name
                for i, arg in enumerate(decorator.args[0].elts):
                    if isinstance(arg, ast.Constant) and arg.value == self.old_name:
                        arg.value = self.new_name
        return self.generic_visit(node)


def replace_branch_destination(
    multiplexer: Callable, old_dest: str, new_dest: str, source: str = None
):
    if source is None:
        source = inspect.getsource(multiplexer)

    # Parse the source code into an AST
    tree = ast.parse(source, mode="exec")

    # Transform the AST to replace return values
    transformer = RewriteBranch(old_dest, new_dest)
    transformed_tree = transformer.visit(tree)

    # Compile the modified AST back into a code object
    code = compile(transformed_tree, filename="<ast>", mode="exec")
    func_globals = multiplexer.__globals__.copy()
    exec(code, func_globals)

    return (
        func_globals[multiplexer.__name__],
        astunparse.unparse(transformed_tree).strip(),
    )
