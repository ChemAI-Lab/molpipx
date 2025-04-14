# To test our main.rs file
run 
```bash
RUSTFLAGS="-Z autodiff=Enable,LooseTypes" cargo +enzyme run --release
```
and pray


# For benchmarking
```
python3 bench.py
```
and wait ~2hrs. Afterwards you'll find a bench.csv with all the data.
