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


def init_monos(Lines, f_out_monomials):
    f_out = open(f_out_monomials, 'a+')
    f_out.write('fn f_init_monomials(mono: &mut [f32; N_MONOS], r: & [f32; N_DISTANCES]) { \n')
    # skip first line and set first entry to 1.0, thus i+1 for all following steps
    f_out.write('  mono[0] = 1.0;\n')
    for i, line in enumerate(Lines[1:]):
        z = line.split()
        z = np.array(z,dtype=int)
        x = z[1:]
        assert(z[0] == 0)
        j = f_monomial_flag_0(x)
        assert(j != -1)
        f_out.write('  mono[{}] = r[{}];\n'.format(i+1,j))
    f_out.write('}\n\n')
    f_out.close()

def gen_mono_block(block, i, offset, f_out_monomials):
    f_out = open(f_out_monomials, 'a+')
    f_out.write('fn f_monomials{}(mono: &mut [f32; N_MONOS]) {{ \n'.format(i))
    for j,line in enumerate(block):
        z = line.split()
        z = np.array(z,dtype=int)
        assert (z[0] == 1)
        a,b = z[1],z[2]
        f_out.write('  mono[{}] = mono[{}] * mono[{}];\n'.format(j+offset,a,b))
    f_out.write('}\n\n')
    f_out.close()


def create_f_monomials(file_mono, file_label):
    
    f_out_monomials = 'monomials_{}.rs'.format(file_label)
    f_out = open(f_out_monomials, 'w+')

    f = open(file_mono, 'r')
    Lines = f.readlines() 
    n_mono = len(Lines)
    index = 0 
    while Lines[index][0] == '0':
        index += 1
    zeros, ones = (Lines[:index], Lines[index:])
    
    f_out.write('// File created from {} \n'.format(file_mono))
    f_out.write('// Total number of monomials = {} \n\n'.format(n_mono))
    f_out.write('pub const N_MONOS: usize = {};\n'.format(n_mono))

    f_out.close()

    block_size = 50
    blocks = [ones[i:i+block_size] for i in range(0, len(ones), block_size)]
    offset = len(zeros)
    for i, block in enumerate(blocks):
        gen_mono_block(block, i, offset, f_out_monomials)
        offset += block_size

#     ----------------------------
#     MAIN
    f_out = open(f_out_monomials, 'a+')
    init_monos(zeros, f_out_monomials)
    f_out.write('pub fn f_monomials(r: &[f32; N_DISTANCES]) -> [f32; N_MONOS] {\n\n')
    f_out.write('  let mut mono = [0.0; N_MONOS];\n\n'.format(n_mono))
    f_out.write('  f_init_monomials(&mut mono, r);\n\n')
    
    for i, block in enumerate(blocks):
        f_out.write('  f_monomials{}(&mut mono);\n'.format(i))
    
    
    f_out.write('\n')
    f_out.write('  return mono; \n')
    f_out.write('}\n')
    f_out.write('\n')
    f_out.write('\n')
    f_out.close()
    f.close() 
# ----------------------------------------------------------------------------------------
#   READ POLYNOMIALS 


def create_poly_block(block, i, offset, f_out_polynomials):
    f_out = open(f_out_polynomials, 'a+')
    f_out.write('fn f_polynomials{}(poly: &mut [f32; N_POLYS],mono: &[f32; N_MONOS]) {{\n'.format(i))
    for i,l in enumerate(block):
        z = l.split()
        z = np.array(z,dtype=int)
        x = z[1:]
#         FLAG = 2
        if z[0] == 2:
            #mono.at[107].set(mono[2] * mono[41]) 
            str_ = '    poly[{}] = '.format(i+offset)
            for j, xi in enumerate(x):
                str_ += 'mono[{}]'.format(xi)
                if j < x.shape[0] -1 :
                    str_ += ' + '
#             str_ += ')'
            f_out.write('{};\n'.format(str_))
#             print(str_)

#         FLAG = 3 
        elif z[0] == 3:
            str_ = '    poly[{}] = '.format(i+offset)
            for j, xi in enumerate(x):
                str_ += 'poly[{}]'.format(xi)
#                 str_ += 'jnp.take(poly,{})'.format(xi)
                if j == 0:
                    str_ += ' * '               
                elif j < x.shape[0]-1:
                    str_ += ' - '
#             str_ += ')'
            f_out.write('{};\n'.format(str_))
    f_out.write('}\n\n')
    f_out.close()



def create_f_polynomials(file_poly,file_label):

    f_out_polynomials = 'polynomials_{}.rs'.format(file_label)
    
    f = open(file_poly,'r')
    Lines = f.readlines() 
    n_poly = len(Lines)
    block_size = 50
    
    f_out = open(f_out_polynomials, 'w+')
    f_out.write('use crate::monomials_{}::*;\n\n\n'.format(file_label))
    f_out.write('pub const N_POLYS: usize = {};\n\n'.format(n_poly))
    f_out.write('// File created from {} \n\n\n'.format(file_poly))
    f_out.close()
    
    blocks = [Lines[i:i+block_size] for i in range(0, len(Lines), block_size)]
    offset = 0
    for i, block in enumerate(blocks):
        create_poly_block(block, i, offset, f_out_polynomials)
        offset += block_size

    f_out = open(f_out_polynomials, 'a+')
    f_out.write('// Total number of monomials = {} \n\n'.format(n_poly))
    f_out.write('pub fn f_polynomials(r: &[f32; N_DISTANCES]) -> [f32; N_POLYS] {\n\n')
    f_out.write('    let mono: [f32; N_MONOS] = f_monomials(r);\n')
    f_out.write('    let mut poly = [0.0; N_POLYS];\n\n'.format(n_poly))
    for i in range(len(blocks)):
        f_out.write('    f_polynomials{}(&mut poly, &mono);\n'.format(i))
    f_out.write('\n')      
    f_out.write('    return poly;\n')
    f_out.write('}\n')
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

    #file_mono = 'data/{}.MONO'.format(f_head)
    #file_poly = 'data/{}.POLY'.format(f_head)

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
    

