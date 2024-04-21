import os

from enum import Enum
from typing import Optional, List, BinaryIO, Tuple, Dict

from warnings import warn
from tempfile import NamedTemporaryFile

from ._utils import aig_to_cnf, exec_abc

Gate = List[int]


class EncFmt(Enum):
    AIG = 'aig'
    AAG = 'aag'
    CNF = 'cnf'


class EncError(Exception):
    pass


def read_delta(pointer: BinaryIO) -> int:
    delta = 0
    i, ch = 0, pointer.read(1)[0]
    while ch & 0x80:
        delta |= (ch & 0x7f) << (7 * i)
        i, ch = i + 1, pointer.read(1)[0]
    return delta | (ch << (7 * i))


def write_delta(pointer: BinaryIO, delta: int):
    while delta & ~0x7f:
        ch = (delta & 0x7f) | 0x80
        pointer.write(bytes([ch]))
        delta >>= 7
    pointer.write(bytes([delta]))


def detect_fmt(filepath: str) -> Optional[EncFmt]:
    _filename = os.path.split(filepath)[-1]
    _file_ext = _filename.split('.')[-1]
    try:
        return EncFmt(_file_ext)
    except ValueError:
        return None


def split_lit(literal: int) -> Tuple[int, int]:
    return literal >> 1, literal % 2


class AIG:
    _max_var = 0

    def __init__(self, from_file: str = None, from_fp: BinaryIO = None):
        """
            Basic constructor. Creates an encoding instance by reading from a
            file by path or binary pointer. If the input parameters are not
            defined, then an empty encoding is created.

            :param from_file: a file path.
            :param from_fp: a binary file pointer.

            :type from_file: Optional(str)
            :type from_fp: Optional(BinaryIO)

            :raises EncError: if the encoding format could not be determined.
        """

        self._gates = []
        self._inputs = []
        self._latches = []
        self._outputs = []
        self._comments = []

        if from_fp:
            self._read_from_fp(from_fp)
        elif from_file:
            self._read_from_file(from_file)

    def _read_from_file(self, filepath: str):
        with open(filepath, 'rb') as pointer:
            self._read_from_fp(pointer)

    def _read_from_fp(self, pointer: BinaryIO):
        header = pointer.readline().decode().split()
        _fmt = header[0] if len(header) > 1 else None

        print(_fmt)
        if _fmt == EncFmt.AIG.value:
            self._read_aig(pointer, header)
        elif _fmt == EncFmt.AAG.value:
            self._read_aag(pointer, header)
        else:
            raise EncError(f'unknown header format: {_fmt}')

    def _read_aig(self, pointer: BinaryIO, header: List[str]):
        self._max_var, _in, _lc, _out, _gates = map(int, header[1:])
        assert _lc == 0, 'Unsupported latches in circuits now'

        for i in range(_in):
            self._inputs.append((i + 1) << 1)

        for i in range(_lc):
            self._latches.append((_in + i + 1) << 1)

        for _ in range(_out):
            var = pointer.readline().strip()
            self._outputs.append(int(var.decode()))
        self._outputs = sorted(self._outputs)

        for i in range(_gates):
            lhs = (_in + _lc + i + 1) << 1
            rhs0 = lhs - read_delta(pointer)
            rhs1 = rhs0 - read_delta(pointer)
            self._gates.append([lhs, rhs0, rhs1])

        for comment in pointer.readlines():
            self._comments.append(comment.decode())

    def _read_aag(self, pointer: BinaryIO, header: List[str]):
        self._max_var, _in, _lc, _out, _gates = map(int, header[1:])
        assert _lc == 0, 'Unsupported latches in circuits now'

        for _ in range(_in):
            var = pointer.readline().decode()
            self._inputs.append(int(var.strip()))
        self._inputs = sorted(self._inputs)

        for _ in range(_lc):
            var = pointer.readline().decode()
            self._latches.append(int(var.strip()))
        self._latches = sorted(self._latches)

        for _ in range(_out):
            var = pointer.readline().decode()
            self._outputs.append(int(var.strip()))
        self._outputs = sorted(self._outputs)

        for _ in range(_gates):
            gate = pointer.readline().decode().strip()
            self._gates.append(list(map(int, gate.split())))

        for comment in pointer.readlines():
            self._comments.append(comment.decode())

    def to_file(self, filepath: str, fmt: EncFmt = None):
        """
            Writes current encoding to file. The encoding format is specified
            using the output file extension or optional input parameter ``fmt``.

            :param filepath: a output file path.
            :param fmt: a enum of encoding format.

            :type filepath: str
            :type fmt: Optional(EncFmt)

            :raises EncError: if the encoding format could not be determined.
        """

        if not isinstance(fmt, EncFmt):
            fmt = detect_fmt(filepath)
        if not isinstance(fmt, EncFmt):
            raise EncError(
                'to_file(...) method unknown output format, use'
                ' <fmt: EncFmt> param or specify file extension'
            )
        with open(filepath, 'wb+') as pointer:
            if fmt == EncFmt.AIG:
                return self._write_aig(pointer)
            elif fmt == EncFmt.AAG:
                return self._write_aag(pointer)
            elif fmt == EncFmt.CNF:
                return self._write_cnf(pointer)

    def _write_aig(self, pointer: BinaryIO):
        pointer.write(
            (f'aig {self._max_var} {len(self._inputs)} {len(self._latches)}'
             f' {len(self._outputs)} {len(self._gates)}\n' + '\n'.join(
                map(str, self._outputs)) + '\n').encode()
        )
        for lhs, rhs0, rhs1 in self.gates:
            write_delta(pointer, lhs - max(rhs0, rhs1))
            write_delta(pointer, abs(rhs0 - rhs1))
        for comment in self._comments:
            pointer.write(comment.encode())

    def _write_aag(self, pointer: BinaryIO):
        pointer.write(
            (f'aag {self._max_var} {len(self._inputs)} {len(self._latches)}'
             f' {len(self._outputs)} {len(self._gates)}\n' + '\n'.join(
                map(str, (*self._inputs, *self._latches, *self._outputs)))
             + '\n').encode()
        )
        for lhs, rhs0, rhs1 in self.gates:
            pointer.write(f'{lhs} {rhs0} {rhs1}\n'.encode())
        for comment in self._comments:
            pointer.write(comment.encode())

    def _write_cnf(self, pointer: BinaryIO):
        cnf = aig_to_cnf(self)
        for comment in cnf.header:
            pointer.write(comment.encode())
        for clause in cnf.clauses:
            str_clause = ' '.join(map(str, clause))
            pointer.write(f'{str_clause} 0\n'.encode())

    def repeat(self, literals: List[int], count: int) \
            -> Tuple['AIG', Dict[int, int]]:
        input_mapping, repeatable = {}, set()
        re_literals = set()
        for var, sign in map(split_lit, literals):
            if var << 1 not in self._inputs:
                warn(f'Not input literal {var << 1} '
                     f'was omitted while repeat')
            else:
                repeatable.add(var)
                re_literals.add(var << 1)

        print(len(repeatable), count)
        if len(repeatable) < 1 or count <= 1:
            inputs, outputs = self.inputs, self._outputs
            gates, max_var = self._gates, self._max_var

            for var, _ in map(split_lit, inputs):
                input_mapping[var] = [var]
        else:
            remain_ins = set(self._inputs) - re_literals
            in_len, re_len = len(remain_ins), len(repeatable)
            max_var = in_len + (self._max_var - in_len) * count

            free_var, gates, outputs = 1, [], []
            for var, _ in map(split_lit, self._inputs):
                if var not in repeatable:
                    input_mapping[var] = free_var
                    free_var += 1

            for var, _ in map(split_lit, self._inputs):
                if var in repeatable:
                    input_mapping[var] = free_var
                    free_var += 1

            gates_len, inputs = len(self._gates), [
                (i + 1) << 1 for i in
                range(in_len + re_len * count)
            ]

            def map_lit(_lit: int, _number: int) -> int:
                _var, _sign = split_lit(_lit)
                if _var in input_mapping:
                    _var = input_mapping[_var]
                    if _var in repeatable:
                        _var += re_len * _number
                else:
                    _var += re_len * (count - 1)
                    _var += gates_len * _number
                return (_var << 1) + _sign

            for number in range(count):
                gates.extend([[
                    map_lit(var, number)
                    for var in gate
                ] for gate in self._gates])
                outputs.extend([
                    map_lit(out, number)
                    for out in self._outputs
                ])

        encoding = AIG()
        encoding._gates = gates
        encoding._inputs = inputs
        encoding._outputs = outputs
        encoding._max_var = max_var
        # todo: understand latches
        encoding._latches = self._latches

        return encoding, input_mapping

    def substitute(self, literals: List[int]) -> Tuple['AIG', Dict[int, int]]:
        print(literals)
        input_mapping, substitution = {}, {}
        for var, sign in map(split_lit, literals):
            if var << 1 not in self._inputs:
                warn(f'Not input literal {var << 1} '
                     f'was omitted while substituting')
            else:
                substitution[var] = sign

        if len(substitution) == 0:
            inputs, gates = self.inputs, self._gates
            for var, _ in map(split_lit, inputs):
                input_mapping[var] = var
        else:
            free_var, inputs, gates = 1, [], []
            for var, _ in map(split_lit, self._inputs):
                if var not in substitution:
                    input_mapping[var] = free_var
                    inputs.append(free_var << 1)
                    free_var += 1

            for var, sign in substitution.items():
                value = 1 - substitution[var]
                input_mapping[var] = free_var
                gates.append([free_var << 1, value, value])
                free_var += 1

            def map_lit(lit: int) -> int:
                _var, _sign = split_lit(lit)
                if _var << 1 in input_mapping:
                    _var = input_mapping[_var]
                return (_var << 1) + _sign

            gates += [
                [map_lit(lit) for lit in gate]
                for gate in self._gates
            ]

        encoding = AIG()
        encoding._gates = gates
        encoding._inputs = inputs
        encoding._outputs = self._outputs
        encoding._latches = self._latches
        # todo: understand latches
        encoding._max_var = self._max_var

        return encoding, input_mapping

    def simplify(self, method: str = 'fraig', verbose: bool = False) -> 'AIG':
        with NamedTemporaryFile(mode='r+b', suffix='.aig') as simp_file:
            with NamedTemporaryFile(delete=False, suffix='.aig') as orig_file:
                self._write_aig(pointer=orig_file)
            try:
                exec_abc([
                    f'read {orig_file.name}',
                    f'{method}',
                    f'write {simp_file.name}'
                ], verbose)
            finally:
                os.remove(orig_file.name)

            return AIG(from_fp=simp_file)

    @property
    def max_var(self) -> int:
        return self._max_var

    @property
    def inputs(self) -> List[int]:
        return list(self._inputs)

    @property
    def latches(self) -> List[int]:
        return list(self._latches)

    @property
    def outputs(self) -> List[int]:
        return list(self._outputs)

    @property
    def gates(self) -> List[Gate]:
        return list(self._gates)


__all__ = [
    'AIG',
    # types
    'EncFmt',
    'EncError'
]
