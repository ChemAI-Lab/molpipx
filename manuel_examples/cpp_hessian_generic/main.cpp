#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <vector>

#include "parse.h"
#include "smart.h"

using std::vector;
typedef vector<double> vd;
typedef vector<int> vi;

double dist(const double *a, const double *b) {
  float dist = std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2) +
               std::pow(a[2] - b[2], 2);
  return std::sqrt(dist);
}

vd dist(const vd positions) {
  size_t entries = (positions.size() / 3);
  vd r((entries * (entries - 1)) / 2);
  size_t pos = 0;
  for (size_t i = 0; i < entries; i++)
    for (size_t j = i + 1; j < entries; j++)
      r[pos++] = dist(&positions[3 * i], &positions[3 * j]);
  return r;
}

vd morse(vd r, float l) {
  for (size_t i = 0; i < r.size(); i++) {
    r[i] = exp(-r[i] / l);
  }
  return r;
}
// __restrict__
// __restrict__
// __restrict__
// __restrict__

void f_energy(const vd &inputs, double &out, const vi &Morse_indices,
              const vi &MONY_vec, size_t M, const vi &POLY_vec, size_t P) {

  const vd distances = dist(inputs);
  vector<double> z = morse(distances, 1.0);

  vector<double> polys =
      create_f_polynomials(z, Morse_indices, MONY_vec, M, POLY_vec, P);

  assert(polys.size() == P);
  for (std::size_t i = 0; i < P; i++) {
    out += polys[i];
  }
}

std::vector<double> morse(vd r, double l) {
  for (size_t i = 0; i < r.size(); i++) {
    r[i] = exp(-r[i] / l);
  }
  return r;
}

/// Enzyme

int enzyme_dup;
int enzyme_const;
// clang-format off
void __enzyme_autodiff(void *, 
                      int, const vd &__restrict__, vd &__restrict__, // shaddow
                      int, double &__restrict__, double &__restrict__, // shaddow
                      int, vi &__restrict__, 
                      int, vi &__restrict__, size_t, 
                      int, const vi &__restrict__, size_t);

void __enzyme_fwddiff2(void *, 
                      int, const vd &, vd &,// x, dx 
                      int, vd &, vd &,// bx, dbx
                      int, double &, double &,// y, dy
                      int, double &, double &,// by, dby
                      int, vi &__restrict__, 
                      int, vi &__restrict__, size_t,
                      int, const vi &__restrict__, size_t);

// nice wrappers jacobians
void jac_rev(const vd &__restrict__ x, vd &__restrict__ bx, 
             double &__restrict__ y, double &__restrict__ by,
             vi &__restrict__ Morse_indices,
             vi &__restrict__ MONY_vec, size_t M,
             const vi &__restrict__ POLY_vec, size_t P) {
  __enzyme_autodiff((void *)f_energy, 
                      enzyme_dup, x, bx, 
                      enzyme_dup, y, by,
                      enzyme_const, Morse_indices, 
                      enzyme_const, MONY_vec, M, 
                      enzyme_const, POLY_vec, P);
}


void hess_fwdrev(const vd &__restrict__ x, vd &__restrict__ dd_x,
             vi &__restrict__ Morse_indices,
             vi &__restrict__ MONY_vec, size_t M,
             const vi &__restrict__ POLY_vec, size_t P) {
  assert(x.size() * x.size() == dd_x.size());
  for (size_t i = 0; i < x.size(); i++) {
    vd bx(x.size(), 0.0);
    vd dx(x.size(), 0.0);
    vd dbx(x.size(), 0.0);

    dx[i] = 1.0;

    double y = 0.0;
    double by = 1.0;
    double dy = 0.0;
    double dby = 0.0;

    __enzyme_fwddiff2((void *)jac_rev, 
        enzyme_dup, x,  dx, 
        enzyme_dup, bx, dbx,
        enzyme_dup, y, dy, 
        enzyme_dup, by, dby,
        enzyme_const, Morse_indices, 
        enzyme_const, MONY_vec, M, 
        enzyme_const, POLY_vec, P);
    for (size_t j = 0; j < x.size(); j++) {
      dd_x[i * x.size() + j] = dbx[j];
    }
  }
}
// clang-format on

int main(int argc, char *argv[]) {
  std::string base;
  if (argc == 1) {
    base = "data/MOL_1_3_4.";
  } else {
    base = argv[1];
  }
  std::cout << "using following path for data: " << base << std::endl;

  // POLY (are parsed correct, verified)
  const vector<int> POLY_vec = parse_poly(base + "POLY");
  size_t P = std::count(POLY_vec.begin(), POLY_vec.end(), -1);

  // MONO (both are correct, verified)
  vector<int> MORSE_indices, MONO_vec;
  std::tie(MORSE_indices, MONO_vec) = parse_mono(base + "MONO");
  size_t M = MORSE_indices.size() + (MONO_vec.size() / 2) + 1;

  std::cout << "M: " << M << "\t"
            << "P: " << P << std::endl;

  vector<double> r;
  // if (M == 33) {
  //   r = r2;
  // } else if (M == 2911) {
  //   r = r1;
  // } else {
  //   return -1;
  // }

  // init some random weights
  const vd x = {0.00000000,  0.00000000, 14.00307401, 0.00000000,
                -1.98144397, 1.00782504, -1.71598081, 0.99072198,
                1.00782504,  1.71598081, 0.99072198,  1.00782504};
  double y = 0.0;
  f_energy(x, y, MORSE_indices, MONO_vec, M, POLY_vec, P);
  std::cout << " ENERGY: " << y << std::endl;

  // vd dx(x.size());
  // y = 0.0;
  // double by = 1.0;
  // jac_rev(x, dx, res, bres, MORSE_indices, MONO_vec, M, POLY_vec, P);
  // std::cout << std::endl << "jac rev: ";
  // for (size_t i = 0; i < dx.size(); i++) {
  //   if (i % 3 == 0)
  //     std::cout << std::endl;
  //   std::cout << dx[i] << " \t";
  // }

  // Hessian - incorrect
  vd dx(x.size());
  vd dd_x(x.size() * x.size());

  hess_fwdrev(x, dd_x, MORSE_indices, MONO_vec, M, POLY_vec, P);

  for (size_t i = 0; i < dd_x.size(); i += 3) {
    if (i % (3 * 4) == 0)
      std::cout << std::endl;
    if (i % (3 * 4 * 3) == 0)
      std::cout << std::endl;
    std::cout << std::setprecision(1) << std::scientific << "[" << dd_x[i]
              << " " << dd_x[i + 1] << " " << dd_x[i + 2] << "]" << std::endl;
  }

  std::cout << std::endl;

  return 1;
}
