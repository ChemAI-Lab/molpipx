## Monomial and polynomial files generator for an ABC<sub>2</sub>D<sub>3</sub>EF molecule

To generate the monomial and polynomial files in a format compatible with JAX and RUST, users can use the `msa_file_generator` function from the MOLPIPx library. Below is an example for Ethanol, represented as ABC<sub>2</sub>D<sub>3</sub>EF, for generating monomial and polynomial functions translated from MSA:
 

 ```python
from molpipx import msa_file_generator

msa_head = 'MOL_1_1_2_3_1_1_2'
label = '1_1_2_3_1_1_2'
path = 'Ethanol/'
msa_file_generator(msa_head,path,label)

```
