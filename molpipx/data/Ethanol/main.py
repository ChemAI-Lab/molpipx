from molpipx import msa_file_generator

def main():
    msa_head = 'MOL_1_1_2_3_1_1_2'
    label = '1_1_2_3_1_1_2'
    path = './'
    msa_file_generator(msa_head,path,label)
    
    
if __name__ == '__main__':
    main()
