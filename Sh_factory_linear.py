import os

def sh_file(config, workdir):
    ###########################################################################>
    ## DEFINE NAME TAIL FOR SCRIPT

    nameTail = f'results_linear_{config.poly_degree}'  # name of the file to save 
    subfolder = os.path.join(workdir, 'scripts_and_logs/LinearPIP')  # specify the subfolder

    ###########################################################################>
    ## CREATE SUBFOLDER IF IT DOESN'T EXIST

    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    ###########################################################################>
    ## OPEN AND WRITE SHELL SCRIPT

    script_path = os.path.join(subfolder, f"{nameTail}.sh")
    log_path = os.path.join(subfolder, f"out_{nameTail}.log")

    with open(script_path, 'w+') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --account=rrg-ravh011\n')
        f.write(f'#SBATCH --job-name={nameTail}\n')
        f.write('#SBATCH --time=01:00:00\n')  # time of computation -- NOTE: HERE, 72 hours.
        f.write(f'#SBATCH --output={log_path}\n')
        f.write(f'#SBATCH --nodes=1\n')
        f.write(f'#SBATCH --ntasks-per-node=40\n')
        # f.write(f'#SBATCH --mem-per-cpu=8000M\n')
        # f.write(f'#SBATCH --cpus-per-task=2\n\n')
        
        ###########################################################################>
        ## LOAD MODULES

        f.write('module load python/3.11\n')  # change based on version
        f.write('source /home/r/ravh011/asmaj/virtual_envs/PIP/bin/activate\n')  # load your environment

        ###########################################################################>
        ## WRITE SCRIPT EXECUTION

        f.write(f'python /scratch/r/ravh011/asmaj/PIPMSA_jax/examples/linear_pip/main.py --workdir {workdir} '
                f'--molecule {config.molecule} '
                f'--poly_degree {config.poly_degree} '
                f'--ntr {config.ntr} '
                f'--nval {config.nval} '
                f'--grad_bool {config.grad_bool}\n')

        f.write('\n\n')
        f.close()

    ###########################################################################>
    ## SUBMIT JOB TO COMPUTE CANADA

    if os.path.isfile(script_path):
        print(f"Submitting {script_path}")
        os.system(f"sbatch {script_path}")

def main():
    from examples.linear_pip.main import merge_configs

    config, workdir, _ = merge_configs()
    poly_degree = [3] 
    for _poly_degree in poly_degree:
        config.poly_degree = _poly_degree
        sh_file(config, workdir)
    # sh_file(config, workdir)

if __name__ == "__main__":
    main()
