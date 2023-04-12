#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::vector<double>
create_f_monomials(const std::vector<double> &__restrict__ Morse,
                   const std::vector<int> &__restrict__ Morse_indices,
                   const std::vector<int> &__restrict__ MONO_ss, size_t M) {
  std::vector<double> res(M);

  std::stringstream ss;
  size_t index = 1;

  res[0] = 1.0;
  for (size_t i = 0; i < Morse_indices.size(); i++) {
    int pos = Morse_indices[i];
    res[index++] = Morse[pos];
  }

  for (size_t i = 0; i < MONO_ss.size(); i += 2) {
    int pos1 = MONO_ss[i];
    int pos2 = MONO_ss[i + 1];
    res[index++] = res[pos1] * res[pos2];
  }

  return res;
}

std::vector<double>
create_f_polynomials(std::vector<double> Morse,
                     const std::vector<int> &__restrict__ Morse_indices,
                     const std::vector<int> &__restrict__ MONO_vec, size_t M,
                     const std::vector<int> &__restrict__ POLY_vec, size_t P) {

  // for (size_t i = 0; i < Morse.size(); i++)
  //   std::cout << Morse[i] << std::endl;
  // std::cout << "were Morses\n";
  std::vector<double> monomials =
      create_f_monomials(Morse, Morse_indices, MONO_vec, M);
  assert(monomials.size() == M);
  // for (size_t i = 0; i < monomials.size(); i++)
  //   std::cout << monomials[i] << std::endl;
  // std::cout << "were monomials\n";
  // for (size_t i = 0; i < POLY_vec.size(); i++)
  //   std::cout << POLY_vec[i] << std::endl;
  // std::cout << "was poly_vec\n";

  std::vector<double> res(P);
  int pos = 0;
  for (size_t i = 0; i < P; i++) {
    if (POLY_vec[pos] == 3) {
      pos++; // don't re-read flag integer
      int pos1 = POLY_vec[pos++];
      int pos2 = POLY_vec[pos++];
      double tmp = res[pos1] * res[pos2];
      while (POLY_vec[pos] != -1) {
        int pos1 = POLY_vec[pos];
        tmp -= res[pos1];
        pos++;
      }
      res[i] = tmp;
    } else {
      pos++; // don't re-read flag integer
      while (POLY_vec[pos] != -1) {
        int index = POLY_vec[pos];
        res[i] += monomials[index];
        pos++;
      }
    }
    pos++; // s.t. we read the next flag next round
  }
  // for (size_t i = 0; i < res.size(); i++)
  //   std::cout << res[i] << std::endl;
  // std::cout << "were polynomials\n";
  return res;
}
