import os

def sh_file(config, workdir):
    ###########################################################################>
    ## DEFINE NAME TAIL FOR SCRIPT

    nameTail = f'results_gp_{config.ntr}_{config.kernel_type}'  # name of the file to save 
    subfolder = os.path.join(workdir, 'scripts_and_logs/PIPGP')  # specify the subfolder

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
        f.write('#SBATCH --time=03:00:00\n')
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

        f.write(f'python /scratch/r/ravh011/asmaj/PIPMSA_jax/examples/pipgp/main.py --workdir {workdir} '
                f'--kernel_type {config.kernel_type} '
                f'--decay_steps {config.decay_steps} '
                f'--peak_value {config.peak_value} '
                f'--warmup_steps {config.warmup_steps} '
                f'--end_value {config.end_value} '
                f'--molecule {config.molecule} '
                f'--poly_degree {config.poly_degree} '
                f'--ntr {config.ntr} '
                f'--ntst {config.ntst} '
                f'--num_iter {config.num_iter}\n')

        f.write('\n\n')
        f.close()

    ###########################################################################>
    ## SUBMIT JOB TO COMPUTE CANADA

    if os.path.isfile(script_path):
        print(f"Submitting {script_path}")
        os.system(f"sbatch {script_path}")

def main():
    from examples.pipgp.main import merge_configs

    config, workdir = merge_configs()
    print('workdir:', workdir)
    # ntr = [200, 400, 600, 800, 1000, 2000] #[1500, 2000]
    ntr = [200]
    # kernels = ['RBF', 'Matern12', 'Matern32', 'Matern52']
    kernels = ['RBF']
    for _ntr in ntr:
        for _kernel in kernels:
            config.ntr = _ntr
            config.kernel_type = _kernel 
            sh_file(config, workdir)
    # sh_file(config, workdir)

if __name__ == "__main__":
    main()
