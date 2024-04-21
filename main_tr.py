from os import makedirs
from os.path import join
from itertools import chain

from enc import AIG

SK_LEN, IV_LEN = 80, 80
OUT_LEN, REPEATS = 1, 100


def to_bin(value: int, size: int):
    return [1 if value & (1 << (size - 1 - i))
            else 0 for i in range(size)]


def get_bits_1to100():
    return '1to100', chain(*[
        to_bin(i + 1, IV_LEN)[::-1]
        for i in range(REPEATS)
    ])


if __name__ == '__main__':
    out_name = f'trivium_576_{REPEATS}ivs_stream1'
    out_dir = join('output', 'trivium_repeat')
    makedirs(out_dir, exist_ok=True)

    filename = 'trivium_init576_stream1.aag'
    encoding = AIG(from_file=join('templates', filename))

    repeat_literals = encoding.inputs[SK_LEN:]
    encoding, _ = encoding.repeat(repeat_literals, REPEATS)
    encoding.to_file(join(out_dir, f'{out_name}.aag'))

    sub_key, bits = get_bits_1to100()
    sub_enc, _ = encoding.substitute([
        var if val else var + 1 for var, val
        in zip(encoding.inputs[SK_LEN:], bits)
    ])

    simp_enc = sub_enc.simplify(verbose=True)
    simp_enc.to_file(join(out_dir, f'{out_name}_{sub_key}_s.aag'))
    simp_enc.to_file(join(out_dir, f'{out_name}_{sub_key}_s.cnf'))
