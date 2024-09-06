# How to compute the linear weights with fortran.

Use DGELSS where A is the training data matrix composed by the PIP expansion (gradients can also be considered). b is the energy (and also gradients). 
https://netlib.org/lapack/explore-html/d7/d3b/group__double_g_esolve_gaa6ed601d0622edcecb90de08d7a218ec.html

Eq. (2) from https://doi.org/10.1021/acs.jctc.9b00043


