   Compiling rust_msa v0.1.0 (/h/292/drehwald/prog/rust_msa)
warning: unused variable: `dx1`
   --> src/main.rs:187:9
    |
187 |     let dx1: Vec<f32> = vec![0.0; x.len()];
    |         ^^^ help: if this is intentional, prefix it with an underscore: `_dx1`
    |
    = note: `#[warn(unused_variables)]` on by default

warning: unused variable: `out`
   --> src/main.rs:188:9
    |
188 |     let out = d_energy_fwd(&x, &mut dx, &weights, 1.0);
    |         ^^^ help: if this is intentional, prefix it with an underscore: `_out`

rustc: /h/292/drehwald/prog/rust/src/llvm-project/llvm/lib/Analysis/ScalarEvolution.cpp:8582: llvm::ScalarEvolution::ExitLimit llvm::ScalarEvolution::computeExitLimitFromSingleExitSwitch(const llvm::Loop*, llvm::SwitchInst*, llvm::BasicBlock*, bool): Assertion `L->contains(Switch->getDefaultDest()) && "Default case must not exit the loop!"' failed.
warning: `rust_msa` (bin "rust_msa") generated 2 warnings
error: could not compile `rust_msa`; 2 warnings emitted

Caused by:
  process didn't exit successfully: `rustc --crate-name rust_msa --edition=2021 src/main.rs --error-format=json --json=diagnostic-rendered-ansi,artifacts,future-incompat --crate-type bin --emit=dep-info,link -C lto=fat -C debuginfo=2 -C metadata=2c89e08b53afa102 -C extra-filename=-2c89e08b53afa102 --out-dir /h/292/drehwald/prog/rust_msa/target/debug/deps -C incremental=/h/292/drehwald/prog/rust_msa/target/debug/incremental -L dependency=/h/292/drehwald/prog/rust_msa/target/debug/deps --extern autodiff=/h/292/drehwald/prog/rust_msa/target/debug/deps/libautodiff-b503fa5af0985bf9.so` (signal: 6, SIGABRT: process abort signal)
