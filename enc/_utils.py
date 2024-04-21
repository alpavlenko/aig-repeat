import os

from pathlib import Path
from subprocess import Popen, PIPE
from warnings import warn
from typing import List, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .aig import AIG


class CNF(NamedTuple):
    header: List[str]
    clauses: List[List[int]]


class Out:
    def __init__(
            self, index: int,
            negate: bool = False
    ):
        self.neg = negate
        self.index = index
        self.offset = None

    def __neg__(self):
        return Out(
            self.index,
            not self.neg
        )

    def get(self, offset: int):
        v = self.index + offset + 1
        return -v if self.neg else v

    def __str__(self):
        raise RuntimeError


def aig_to_cnf(aig: 'AIG') -> CNF:
    clauses, variables = [], {
        var // 2: i + 1 for i, var
        in enumerate(aig.inputs)
    }

    uniq_outputs = set(aig.outputs)
    out_variables = {
        var // 2: Out(i) for i, var in
        enumerate(sorted(uniq_outputs))
    }

    def _get_rhs_var(rhs) -> int:
        return (-variables[rhs // 2] if rhs % 2 else
                variables[rhs // 2]) if rhs > 1 else None

    inputs = sorted(variables.values())
    for lhs, rhs0, rhs1 in aig.gates:
        if lhs // 2 in out_variables:
            var_lhs = out_variables[lhs // 2]
        else:
            var_lhs = len(variables) + 1

        if rhs0 <= 1 or rhs1 <= 1:
            warn('encoding contains 0 or 1 gates,'
                 ' use simplify() before convert!')

        variables[lhs // 2] = var_lhs
        var0 = _get_rhs_var(rhs0)
        var1 = _get_rhs_var(rhs1)

        if rhs0 != 1:
            clauses.append(
                [-var_lhs, var0] if
                var0 else [-var_lhs]
            )
        if rhs1 != 1:
            clauses.append(
                [-var_lhs, var1] if
                var1 else [-var_lhs]
            )
        if rhs0 != 0 and rhs1 != 0:
            clauses.append(list(filter(None, [
                var_lhs, -var0 if var0 else None,
                -var1 if var1 else None
            ])))

    offset = len(variables)
    clauses = [[
        lit.get(offset) if
        isinstance(lit, Out)
        else lit for lit in clause
    ] for clause in clauses]

    outputs = [
        i + offset + 1 for i in
        range(len(out_variables))
    ]
    var_len = offset + len(outputs)
    header = [
        f'cnf {var_len} {len(clauses)}\n',
        f'c inputs: {sorted(inputs)}\n'
        f'c outputs: {sorted(outputs)}\n'
    ]
    return CNF(header=header, clauses=clauses)


def abc_path() -> str:
    root = Path(__file__).absolute().parent.parent
    abc_exec = os.path.join(root, 'abc', 'abc')
    assert os.path.exists(abc_exec), \
        'Compile ABC to use simplify method'
    return abc_exec


def exec_abc(commands: List[str], verbose: bool):
    _input = '\n'.join(commands)
    _output, _errors = Popen(
        [abc_path()], stdin=PIPE,
        stdout=PIPE, stderr=PIPE
    ).communicate(_input.encode())

    if verbose:
        if len(_output) > 0:
            print('--- ABC output ---')
            print(_output.decode().strip())
        if len(_errors) > 0:
            print('--- ABC errors ---')
            print(_errors.decode().strip())
        if len(_output) > 0 or len(_errors) > 0:
            print('--- end of ABC ---')


__all__ = [
    'exec_abc',
    'aig_to_cnf',
    # types
    'CNF'
]
