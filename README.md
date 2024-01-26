# MSA PIP ENZYME MAGIC!

## How to run?
Get your own rust-enzyme fork from your local compiler vendor
Run the following three commands:
```
export ENZYME_STRICT_ALIASING=1
python3 src/generator.py --file /u/drehwald/prog/msa_code/src/data/MOL_2_2_2_2_1_1_1_1_3 --label ethanol
cargo +enzyme run --release
```

Adjust the file and label on the second line based on your own data as you whish.
Enjoy!


when training with energies do not change the sing of the foreces, keep it possitive.

# Generate the files: #
Download the data from [QM-22](https://github.com/jmbowma/QM-22)
| molecule | max polynomial order |	symmetry| Files |
| -------- | -------- | -------- | ------- |
Methane	 | 6 | 4 1 | PIP (deg 3 to 6)
Acetaldehyde |	5 |	4 2 1 or 3 1 1 1 1 | PIP done(4 2 1)
Ethanoal |	3 or 4 |	1 1 2 3 1 1  or 2 5 1 1 | PIP done(1 1 2 3 1 1)
Formic acid dimer |	3 or 4 | 4 4 2
Glycine | 4 | 1 2 1 2 1 2 1
H2CO / HCOH | 7 |	2 1 1
Hydronium |	7 |	3 1
N-methylacetamide | 3 |	3 1 1 1 1 1 1 3
OCHCO cation | 5 | 2 2 1
Tropolone | 3 |1 2 2 2 2 2 2 1 1 or 1 2 6 4 1 1
syn-CH3CHOO (Criegee) |	5 |	4 2 2 or 3 1 1 1 2

