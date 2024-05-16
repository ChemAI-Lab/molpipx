import os
import time
import numpy as np

    

def sh_file(l0:float,bool_wgrads:True)->None:
    f_tail  =  f"l0_{l0:.3f}"

    file_sh = f"pipnn_{f_tail}.sh"
    f=open(file_sh, 'w+')

    f.write('#!/bin/bash \n')
    f.write('#SBATCH --nodes=1 # number of nodes \n')
    f.write('#SBATCH --ntasks=1 # you always get a full node anyway, so why not just try to use everything \n')
    f.write('#SBATCH --account=rrg-ravh011\n')
    f.write('#SBATCH --job-name={} \n'.format(f_tail))
    f.write('#SBATCH--time=1:30:0\n')
    f.write('#SBATCH --output=out_{}.log \n'.format(f_tail))

    f.write('\n')

#     LOAD MODULES
    f.write('source $HOME/.virtualenvs/pipjax/bin/activate \n')

    f.write('\n')

    workdir = '/scratch/r/ravh011/ravh011/pipjax_test/pipnn_results'
    ntr = 1000
    nval = 2500    
    f.write('cd /home/r/ravh011/ravh011/PIPMSA_jax/examples/pipnn \n')
    f.write(f"python main.py --workdir {workdir} --ntr {ntr} --nval {nval} --l0 {l0:.4f} --grad_bool {bool_wgrads}\n")

    f.write('\n')
    f.write('\n')
    f.close()

    if os.path.isfile(file_sh):
        print(f"Submitting {file_sh}")
        # os.system(f"SLURM_MPI_TYPE=pmix_v3 sbatch {file_sh}")
        os.system(f"sbatch {file_sh}")

def main_single_test():
    molecule = 'H2'
    geom_id = 1
    sh_file( molecule, geom_id)


        
        
def main():
    
    # l0 = np.logspace(-4,10,20)
    l0_ = np.array([1E-4,1E-3,1E-2,1E-2,5E-2,1E-1,1,2,5,10],dtype=np.float32)

    for l0 in l0_[:1]:
        sh_file(l0,True)

if __name__ == '__main__':
    main()
    #main_5_random_test()
    # main_lasse_missing_data()
