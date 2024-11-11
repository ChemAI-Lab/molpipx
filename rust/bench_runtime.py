import fileinput
import os
import subprocess
import tempfile

l_placeholder = "pub use crate::data::{molecule}::{{monomials_{number}::*, polynomials_{number}::*}};"
clean = "cargo clean -q"
cmd = "RUSTFLAGS=\"-Z autodiff=LooseTypes\" cargo +enzyme bench --bench pip"
ddir = os.path.join("src", "data")

# collect all molecules in the data folder:
molecules = []
for molecule in os.listdir(ddir):
    if os.path.isdir(os.path.join(ddir, molecule)):
        molecules.append(molecule)
molecules.sort()
print(molecules)

if os.path.isfile("runtime_bench.csv"):
    os.remove("runtime_bench.csv")
for molecule in molecules:
    # now get all degrees in that folder:
    degrees = []
    for degree in os.listdir(os.path.join(ddir, f"{molecule}")):
        if degree.startswith("monomials_"):
            deg = degree.split("monomials_")[1].split(".rs")[0]
            degrees.append(deg)
    degrees.sort()
    print(molecule)
    print(degrees)
    
    # Now create data/mod.rs file and write `pub mod molecule_{};` into it 
    with open(os.path.join(ddir, "mod.rs"), "w") as f:
        f.write("pub mod {};".format(molecule))

    for degree in degrees:
        for line in fileinput.input(os.path.join("src", "lib.rs"), inplace=True):
            if fileinput.filelineno() != 6:
                print(line, end='')
                continue
            new_line = l_placeholder.format(molecule = molecule, number = degree)
            print(new_line)
        
        # Now create data/mod.rs file and write `pub mod molecule_{};` into it 
        with open(os.path.join(ddir, f"{molecule}", "mod.rs"), "w") as f:
            f.write("pub mod monomials_{};".format(degree))
            f.write("pub mod polynomials_{};".format(degree))
        
        # Now bench that new lib.rs file. 
        fp = tempfile.TemporaryFile()
        with tempfile.TemporaryFile(mode='w+') as f:
            with open ("runtime_bench.csv", "a") as csv:
                subprocess.run([cmd], shell=True, check=True, stdout=f)
                csv.write(f"{molecule},{degree}\n")
                f.seek(0)
                for i in range(5):
                    f.readline()
                for line in f:
                    csv.write(line)
                #l = f.readline()
                #time = l.split(" in ")[1].split("s")[0]
                #print(time)


# pub use crate::data::molecule_A2B::{monomials_2_1_3::*, polynomials_2_1_3::*};
# RUSTFLAGS="-Z autodiff=LooseTypes" cargo +enzyme build --release
