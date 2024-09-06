import os
import shutil
import subprocess

# 1) Create a list of all folders in ../pipx/msa_files
# 2) For each folder, unzip the files into the respective data folder 

# Get list of all folders in ../pipx/msa_3_2_files
folders = os.listdir("../pipx/msa_files")
assert len(folders) == 15, "I did not find the expected 15 folders in ../pipx/msa_files"

# if the data folder isn't empty, print an info and exit
if os.listdir("data"):
    shutil.rmtree("data")
    os.mkdir("data")
    print("Data folder is not empty. Emptying it.")


for folder in folders:
    # Unzip files into the respective data folder
    cmd = f"unzip -j -qq ../pipx/msa_files/{folder}/*_files.zip -d data/{folder} -x \"*.BAS\" -x \"*.FOC\" -x \"*.MAP\""
    print("running: ", cmd)
    subprocess.run(["unzip", "-j", "-qq", f"../pipx/msa_files/{folder}/*_files.zip", "-d", f"data/{folder}", "-x", "*.BAS", "-x", "*.FOC", "-x", "*.MAP"], stderr=subprocess.DEVNULL)

print("Unzipped all files")

# python3 src/generator.py --file data/molecule_A2B/MOL_2_1_3 --label 2_1_3 --dir src/data/molecule_A2B

# for folder in folders:
#    call the generator for each MONO/POLY combi in the folder 
for folder in folders:
    files = os.listdir(f"data/{folder}")
    for file in files:
        if file.endswith(".MONO"):
            # assert the POLY file also exists:
            assert file.replace(".MONO", ".POLY") in files, f"POLY file for {file} does not exist"
            # label is everything after the first _ and before the dot `.`:
            # dont use a single split bc. it will loose things afte rthe second _ 
            label = file.split("_", 1)[1]
            label = label.split(".")[0]
            file = file.split(".")[0]
    
            command = f"python3 src/generator.py --file data/{folder}/{file} --label {label} --dir src/data/{folder}"
            print("running: ", command)
            subprocess.run(["python3", "src/generator.py", "--file", f"data/{folder}/{file}", "--label", label, "--dir", f"src/data/{folder}"])#, stderr=subprocess.DEVNULL

print("Generated all data")

# 3) Now we have the .rs files, so we can compile them using cargo +enzyme build --release
#    For that we however need to specify a correct mod.rs 

path = "src/data"
folders = os.listdir(path)
with open("src/data/mod.rs", "w") as f:
    for folder in folders:
        if folder == "mod.rs":
            continue
        f.write(f"pub mod {folder};\n")
