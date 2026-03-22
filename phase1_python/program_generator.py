import ast
import random
from dataclasses import dataclass

@dataclass
class ProgramSpec:
    complexity: int        # 1, 2, or 3
    seed: int
    source: str           # the generated Python source


def generate_program(complexity: int, seed: int) -> ProgramSpec:
    """
    Generate a random Python program at the given complexity level.
    Deterministic given the same seed.
    Incorporates "Anti-Memorization" Controls: Data dependency chains and early branching.
    """
    rng = random.Random(seed)
    
    # Simple AST generation for small valid programs using data dependency logic
    body = []
    vars_in_scope = []
    
    # Configuration based on complexity
    max_vars = complexity + 2
    max_lines = complexity * 5
    num_lines = rng.randint(2, max_lines)
    
    # 1. Initialize variables (Canonicalizing State)
    for i in range(rng.randint(2, 3)):
        var_name = f"var_{i}"
        val = rng.randint(-15, 255) # Use wide range of values
        body.append(ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Constant(value=val),
            lineno=0
        ))
        vars_in_scope.append(var_name)
    
    # "Dead Code" filter: Data dependency chains. Every instruction should use past variables.
    for i in range(num_lines):
        if not vars_in_scope:
            break
            
        op_choice = rng.choice(['assign', 'add', 'sub', 'mul'])
        target_var = rng.choice(vars_in_scope)
        
        # Ensure we sometimes do conditionals to avoid Constant PC Trap
        if complexity > 1 and rng.random() > 0.7:
            cmp_op = ast.Gt() if rng.random() > 0.5 else ast.Lt()
            left_var = rng.choice(vars_in_scope)
            right_val = rng.randint(0, 100)
            
            if_body = [ast.Assign(
                targets=[ast.Name(id=target_var, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=target_var, ctx=ast.Load()), op=ast.Add(), right=ast.Constant(value=1)),
                lineno=0
            )]
            
            body.append(ast.If(
                test=ast.Compare(left=ast.Name(id=left_var, ctx=ast.Load()), ops=[cmp_op], comparators=[ast.Constant(value=right_val)]),
                body=if_body,
                orelse=[],
                lineno=0
            ))
            continue

        # Standard data dependent operations
        if op_choice == 'assign':
            val = rng.randint(-10, 50)
            node = ast.Assign(targets=[ast.Name(id=target_var, ctx=ast.Store())], value=ast.Constant(value=val), lineno=0)
        else:
            op_map = {'add': ast.Add(), 'sub': ast.Sub(), 'mul': ast.Mult()}
            src_var = rng.choice(vars_in_scope)
            node = ast.Assign(
                targets=[ast.Name(id=target_var, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=target_var, ctx=ast.Load()), op=op_map[op_choice], right=ast.Name(id=src_var, ctx=ast.Load())),
                lineno=0
            )
        body.append(node)
        
    # Add a final print to have an output
    body.append(ast.Expr(value=ast.Call(
        func=ast.Name(id="print", ctx=ast.Load()),
        args=[ast.Name(id=vars_in_scope[0], ctx=ast.Load())],
        keywords=[],
        lineno=0
    ), lineno=0))
    
    # Fix missing attributes for AST
    ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    source = ast.unparse(ast.Module(body=body, type_ignores=[]))
    
    return ProgramSpec(complexity=complexity, seed=seed, source=source)
