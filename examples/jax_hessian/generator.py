import os
import numpy as np
import argparse

# ----------------------------------------------------------------------------------------
#   READ MONOMIALS  
def f_monomial_flag_0(x):
    if np.sum(x) > 0:
        j = np.where(x==1)[0]
        j = j[0]
    else:
        j = -1
    return j
 
def create_f_monomials(file_mono,file_label):
    
    f_out_monomials = 'monomials_{}.py'.format(file_label)
    
    f_out = open(f_out_monomials, 'w+')
    f_out.write('import jax \n')
    f_out.write('import jax.numpy as jnp \n')
    f_out.write('\n')
    f_out.write('# File created from {} \n'.format(file_mono))
    f_out.write('\n')
    f_out.close()

    f = open(file_mono, 'r')
    Lines = f.readlines() 
    n_mono = len(Lines)

#     ----------------------------
#     MAIN
    f_out = open(f_out_monomials, 'a+')
    f_out.write('# Total number of monomials = {} \n'.format(n_mono))
    f_out.write('\n')
    f_out.write('def f_monomials(r): \n')
    f_out.write('\n')
    f_out.write('    mono = jnp.zeros({}) \n'.format(n_mono))
    f_out.write('\n')
    
    for i,l in enumerate(Lines):
        z = l.split()
        z = np.array(z,dtype=int)
        x = z[1:]
        
#         FLAG = 0
        if z[0] == 0:
            j = f_monomial_flag_0(x)     
            if j > -1:
#               x.at[idx].set(y)
                f_out.write('    mono_{} = jnp.take(r,{}) \n'.format(i,j))
            else:
                f_out.write('    mono_{} = 1. \n'.format(i))
#         FLAG = 1              
        elif int(z[0]) == 1:
            a,b = x[0],x[1]
            f_out.write('    mono_{} = mono_{} * mono_{} \n'.format(i,a,b))
    
    f_out.write('\n')
    f_out.write('#    stack all monomials \n')  
    f_out.write('    mono = jnp.stack([')  
    for i,_ in enumerate(Lines[:]):
        f_out.write('    mono_{},'.format(i))
        if i%5 == 0 and i > 0:
            f_out.write(' \n'.format(i))
             
    f_out.write('    ]) \n')       
    f_out.write('\n')      
    f_out.write('    return mono \n')
    f_out.write('\n')
    f_out.write('\n')
    f_out.write('\n')
    f_out.close()
    f.close() 
# ----------------------------------------------------------------------------------------
#   READ POLYNOMIALS 

def create_f_polynomials(file_poly,file_label):

    f_out_polynomials = 'polynomials_{}.py'.format(file_label)
    
    f_out = open(f_out_polynomials, 'w+')
    f_out.write('import jax \n')
    f_out.write('import jax.numpy as jnp \n')
    f_out.write('\n')
    f_out.write('from monomials_{} import f_monomials as f_monos \n'.format(file_label))
    f_out.write('\n')
    f_out.write('\n')
    f_out.write('# File created from {} \n'.format(file_poly))
    f_out.write('\n')
    f_out.write('\n')
    f_out.close()
    
    f = open(file_poly,'r')
    Lines = f.readlines() 
    n_poly = len(Lines)
    
#     TEST
#     MAIN
    f_out = open(f_out_polynomials, 'a+')
    f_out.write('# Total number of monomials = {} \n'.format(n_poly))
    f_out.write('\n')
    f_out.write('def f_polynomials(r): \n')
    f_out.write('\n')
    f_out.write('    mono = f_monos(r.ravel()) \n'.format(n_poly))
    f_out.write('\n')
    f_out.write('    poly = jnp.zeros({}) \n'.format(n_poly))
    f_out.write('\n')

    for i,l in enumerate(Lines):
        z = l.split()
        z = np.array(z,dtype=int)
        x = z[1:]
#         FLAG = 2
        if z[0] == 2:
            #mono.at[107].set(mono[2] * mono[41]) 
            str_ = '    poly_{} = '.format(i)
            for j, xi in enumerate(x):
                str_ += 'jnp.take(mono,{})'.format(xi)
                if j < x.shape[0] -1 :
                    str_ += ' + '
#             str_ += ')'
            f_out.write('{} \n'.format(str_))
#             print(str_)

#         FLAG = 3 
        elif z[0] == 3:
            str_ = '    poly_{} = '.format(i)
            for j, xi in enumerate(x):
                str_ += 'poly_{}'.format(xi)
#                 str_ += 'jnp.take(poly,{})'.format(xi)
                if j == 0:
                    str_ += ' * '               
                elif j < x.shape[0]-1:
                    str_ += ' - '
#             str_ += ')'
            f_out.write('{} \n'.format(str_))

#     ----------------------- 
    f_out.write('\n')
    f_out.write('#    stack all polynomials \n')  
    f_out.write('    poly = jnp.stack([')  
    for i,_ in enumerate(Lines[:]):
        f_out.write('    poly_{},'.format(i))
        if i%5 == 0 and i > 0:
            f_out.write(' \n'.format(i))               

    f_out.write('    ]) \n')       
#     -----------------------
    f_out.write('\n')      
    f_out.write('    return poly \n')
    f_out.write('\n')
    f_out.write('\n')
    f_out.write('\n')
    f_out.close()
    f.close() 
        


# ----------------------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(description='MSA monomials and polynomials to JAX')
    parser.add_argument('--file', type=str, default='MOL_1_3_4', help='head of the file')
    parser.add_argument('--label', type=str, default='', help='label for the file')
    args = parser.parse_args()
    
    f_head = args.file
    f_label = args.label

    file_mono = '{}.MONO'.format(f_head)
    file_poly = '{}.POLY'.format(f_head)

    if not os.path.isfile(file_mono):
        print('File {} does not exist!'.format(file_mono))
        assert 0
    if not os.path.isfile(file_poly):
        print('File {} does not exist!'.format(file_poly))
        assert 0    

#     construct monomials
    create_f_monomials(file_mono,f_label)

#     construct polynomials
    create_f_polynomials(file_poly,f_label)
    

if __name__ == "__main__":

    main()
    

