# AIG Repeat

[comment]: <> (...)

## Installation

```shell script
git clone git@github.com:alpavlenko/aig-repeat.git
pip install -r requirements.txt
```

To use simplify(...) method, compile ABC:

```shell script
cd abc
make
```

If you have problems with the compilation, then visit the
original [repository](https://github.com/berkeley-abc/abc).

## Usage

Read AIG from file in binary (.aig) and ascii (.aag) format:

```python
from enc import AIG

enc_from_ascii = AIG(from_file='file.aag')
enc_from_binary = AIG(from_file='file.aig')
```

Write AIG to file in binary, ascii format, or convert it to CNF:

```python
from enc import AIG

enc = AIG(from_file='input.aig')
enc.to_file('file_in_ascii.aag')
enc.to_file('file_in_binary.aig')
# 
enc.to_file('file_in_dimacs.cnf')
```

Repeat AIG around subset of input variables:

```python
from enc import AIG

enc = AIG(from_file='file.aig')
repeat_literals = enc.inputs[80:]
# repeat enc 10 times around first 80 input bits
new_enc, _ = enc.repeat(repeat_literals, count=10)
# check length of outputs for new enc 
assert 10 * len(enc.outputs) == len(new_enc.outputs)
```

[comment]: <> (... &#40;add explanatory images&#41;)

Simplify AIG using ABC tool:

```python
from enc import AIG

enc = AIG(from_file='input.aig')
simp_enc = enc.simplify()
simp_enc.to_file('simplified.aig')
```

Substitute values to AIG variables:

```python
from enc import AIG

enc = AIG(from_file='file.aig')
substitution = [
    var if val else var + 1 for var, val
    in zip(enc.inputs[80:], [0] * 80)
]
# substitute 80 zeros into enc input
sub_enc, _ = enc.substitute(substitution)
sub_enc.simplify().to_file('simplified.aig')
```

[comment]: <> (## Examples)

[comment]: <> (...)

