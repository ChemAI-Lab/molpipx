#include <assert.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

using std::vector;
typedef vector<double> vd;

// Functions to fill the vector

vector<double> f_monomials(const vector<double> r) {
  vector<double> mono(33);

  mono[0] = 1.0;
  mono[1] = r[5];
  mono[2] = r[4];
  mono[3] = r[3];
  mono[4] = r[2];
  mono[5] = r[1];
  mono[6] = r[0];

  mono[7] = mono[1] * mono[2];
  mono[8] = mono[1] * mono[3];
  mono[9] = mono[2] * mono[3];
  mono[10] = mono[3] * mono[4];
  mono[11] = mono[2] * mono[5];
  mono[12] = mono[1] * mono[6];
  mono[13] = mono[4] * mono[5];
  mono[14] = mono[4] * mono[6];
  mono[15] = mono[5] * mono[6];
  mono[16] = mono[1] * mono[9];
  mono[17] = mono[1] * mono[10];
  mono[18] = mono[2] * mono[10];
  mono[19] = mono[1] * mono[11];
  mono[20] = mono[3] * mono[11];
  mono[21] = mono[2] * mono[12];
  mono[22] = mono[3] * mono[12];
  mono[23] = mono[2] * mono[13];
  mono[24] = mono[3] * mono[13];
  mono[25] = mono[1] * mono[14];
  mono[26] = mono[3] * mono[14];
  mono[27] = mono[1] * mono[15];
  mono[28] = mono[2] * mono[15];
  mono[29] = mono[4] * mono[15];
  mono[30] = mono[2] * mono[24];
  mono[31] = mono[1] * mono[26];
  mono[32] = mono[1] * mono[28];

  return mono;
}

const std::size_t N = 51;

void f_polynomials0(vector<double> &__restrict__ poly,
                    const vector<double> &__restrict__ mono) {
  poly[0] = mono[0];
  poly[1] = mono[1] + mono[2] + mono[3];
  poly[2] = mono[4] + mono[5] + mono[6];
  poly[3] = mono[7] + mono[8] + mono[9];
  poly[4] = mono[10] + mono[11] + mono[12];
  poly[5] = poly[1] * poly[2] - poly[4];
  poly[6] = mono[13] + mono[14] + mono[15];
  poly[7] = poly[1] * poly[1] - poly[3] - poly[3];
  poly[8] = poly[2] * poly[2] - poly[6] - poly[6];
  poly[9] = mono[16];
  poly[10] = mono[17] + mono[18] + mono[19] + mono[20] + mono[21] + mono[22];
  poly[11] = poly[2] * poly[3] - poly[10];
  poly[12] = mono[23] + mono[24] + mono[25] + mono[26] + mono[27] + mono[28];
  poly[13] = poly[1] * poly[6] - poly[12];
  poly[14] = mono[29];
  poly[15] = poly[1] * poly[3] - poly[9] - poly[9] - poly[9];
  poly[16] = poly[1] * poly[4] - poly[10];
  poly[17] = poly[2] * poly[7] - poly[16];
  poly[18] = poly[2] * poly[4] - poly[12];
  poly[19] = poly[1] * poly[8] - poly[18];
  poly[20] = poly[2] * poly[6] - poly[14] - poly[14] - poly[14];
  poly[21] = poly[1] * poly[7] - poly[15];
  poly[22] = poly[2] * poly[8] - poly[20];
  poly[23] = poly[9] * poly[2];
  poly[24] = mono[30] + mono[31] + mono[32];
  poly[25] = poly[3] * poly[6] - poly[24];
  poly[26] = poly[14] * poly[1];
  poly[27] = poly[9] * poly[1];
  poly[28] = poly[3] * poly[4] - poly[23];
  poly[29] = poly[1] * poly[10] - poly[23] - poly[28] - poly[23];
  poly[30] = poly[1] * poly[11] - poly[23];
  poly[31] = poly[1] * poly[12] - poly[25] - poly[24] - poly[24];
  poly[32] = poly[1] * poly[13] - poly[25];
  poly[33] = poly[4] * poly[5] - poly[25] - poly[31];
  poly[34] = poly[2] * poly[11] - poly[25];
  poly[35] = poly[4] * poly[6] - poly[26];
  poly[36] = poly[2] * poly[12] - poly[26] - poly[35] - poly[26];
  poly[37] = poly[2] * poly[13] - poly[26];
  poly[38] = poly[14] * poly[2];
  poly[39] = poly[3] * poly[3] - poly[27] - poly[27];
  poly[40] = poly[3] * poly[7] - poly[27];
  poly[41] = poly[1] * poly[16] - poly[28];
  poly[42] = poly[2] * poly[21] - poly[41];
  poly[43] = poly[1] * poly[18] - poly[33];
  poly[44] = poly[7] * poly[8] - poly[43];
  poly[45] = poly[6] * poly[6] - poly[38] - poly[38];
  poly[46] = poly[2] * poly[18] - poly[35];
  poly[47] = poly[1] * poly[22] - poly[46];
  poly[48] = poly[6] * poly[8] - poly[38];
  poly[49] = poly[1] * poly[21] - poly[40];
  poly[50] = poly[2] * poly[22] - poly[48];
}

// Total number of monomials = 51

vector<double> f_polynomials(const vector<double> r) {

  const vector<double> mono = f_monomials(r);
  vector<double> poly(51);

  f_polynomials0(poly, mono);

  return poly;
}

/// Primary function(s)

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

void f_energy(const vd &__restrict__ inputs, double &res) {

  const vd weights(N, 1.0);

  const vd distances = dist(inputs);
  const vd z = morse(distances, 1.0);
  vd outs = f_polynomials(z);

  assert(outs.size() == weights.size());

  //double res = 0.0;
  for (std::size_t i = 0; i < N; i++) {
    res += weights[i] * outs[i];
  }
  //return res;
}

/// Enzyme

int enzyme_dup;
int enzyme_const;
double __enzyme_autodiff(void *, int, const vd &, vd &, int, double &,
                         double &);
double __enzyme_fwddiff(void*,int, const vd &, vd &);

// hessian
// clang-format off
void __enzyme_fwddiff2(void *, 
                      int, const vd &, vd &, 
                      int, vd &, vd &, 
                      int, double &, double &,
                      int, double &, double &);
// clang-format on

// nice wrappers jacobians
void jac_rev(const vd &x, vd &bx, double &y, double &by) {
  __enzyme_autodiff((void *)f_energy, enzyme_dup, x, bx, enzyme_dup, y, by);
}

void hess_fwdfwd(const vd &x, vd &dd_x) {
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

    __enzyme_fwddiff2((void *)jac_rev, enzyme_dup, x, dx, enzyme_dup, bx, dbx,
                      enzyme_dup, y, dy, enzyme_dup, by, dby);
    for (size_t j = 0; j < x.size(); j++) {
      dd_x[i * x.size() + j] = dbx[j];
    }
  }
}


int main(int argc, char *argv[]) {

  const vd x = {0.00000000,  0.00000000, 14.00307401, 0.00000000,
                -1.98144397, 1.00782504, -1.71598081, 0.99072198,
                1.00782504,  1.71598081, 0.99072198,  1.00782504};

  //vd weights(N, 1.0);

  // Primary - correct
  double energy = 0.0;
  f_energy(x, energy);
  std::cout << "Energy: " << energy << std::endl;

  // Jacobian - correct
  vd dx(x.size());
  vd dx1(x.size());
  //jac_fwd(x, dx);
  //jac_rev(x, dx1);
  //std::cout << std::endl << "jac fwd: ";
  //for (size_t i = 0; i < dx.size(); i++) {
  //  if (i % 3 == 0)
  //    std::cout << std::endl;
  //  std::cout << dx[i] << " \t";
  //}
  //std::cout << std::endl << std::endl << "jac rev: ";
  //for (size_t i = 0; i < dx1.size(); i++) {
  //  if (i % 3 == 0)
  //    std::cout << std::endl;
  //  std::cout << dx1[i] << " \t";
  //}
  std::cout << std::endl;

  // Hessian - incorrect
  vd dd_x(x.size() * x.size());
  hess_fwdfwd(x, dd_x);

  for (size_t i = 0; i < dd_x.size(); i+=3) {
    if (i % (3*4) == 0)
      std::cout << std::endl;
    if (i % (3*4*3) == 0)
      std::cout << std::endl;
    std::cout << std::setprecision(1) << std::scientific << "[" << dd_x[i] << " " <<dd_x[i+1] << " " << dd_x[i+2] << "]" << std::endl;
  }

  std::cout << std::endl;
  return 1;
}
