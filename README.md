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
