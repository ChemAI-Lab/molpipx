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

double f_energy(const vd &__restrict__ inputs, const vd &__restrict__ weights) {

  const vd distances = dist(inputs);
  const vd z = morse(distances, 1.0);
  vd outs = f_polynomials(z);

  assert(outs.size() == weights.size());

  double res = 0.0;
  for (std::size_t i = 0; i < N; i++) {
    res += weights[i] * outs[i];
  }
  return res;
}

/// Enzyme

int enzyme_dup;
int enzyme_const;
double __enzyme_autodiff(double (*)(const vd &, const vd &), int, const vd &,
                         vd &, int, const vd &);
// typedef struct {
//   double da, db, dc, dd;
// } Ret4;
double __enzyme_fwddiff(double (*)(const vd &, const vd &), const vd &, vd &,
                        int, const vd &);

double __enzyme_fwddiff2(void (*)(const vd &, vd &, const vd &),
                         /* const r */ int, const vd &,
                         /* active d_r*/ vd &, vd &,
                         /* const weights*/ int, const vd &);
double __enzyme_fwddiff3(void (*)(const vd &, vd &, const vd &),
                         /* dup r */ int, const vd &, const vd &,
                         /* active d_r*/ vd &, vd &,
                         /* const weights*/ int, const vd &);
// nice wrappers jacobians
void jac_rev(const vd &r, vd &d_r, const vd &weights) {
  __enzyme_autodiff(f_energy, enzyme_dup, r, d_r, enzyme_const, weights);
}

void jac_fwd(const vd &r, vd &d_r, const vd &weights) {
  for (size_t i = 0; i < r.size(); i++) {
    vd activity(r.size());
    activity[i] = 1.0;
    d_r[i] = __enzyme_fwddiff(f_energy, r, activity, enzyme_const, weights);
  }
}

void hess_fwdfwd2(const vd &r, vd &dd_r, const vd &weights) {
  assert(r.size() * r.size() == dd_r.size());
  for (size_t i = 0; i < r.size(); i++) {
    vd activity(r.size(), 0.0);
    vd out(r.size(), 0.0);
    activity[i] = 1.0;
    __enzyme_fwddiff2(jac_fwd, enzyme_const, r, out, activity, enzyme_const,
                      weights);
    for (size_t j = 0; j < r.size(); j++) {
      dd_r[i * r.size() + j] = out[j];
    }
  }
}

// nice wrappers hessians
void hess_fwdfwd(const vd &r, vd &d_r, vd &dd_r, const vd &weights) {
  vd activity(r.size(), 1.0);
  vd activity2(r.size(), 1.0);
  for (size_t i = 0; i < r.size(); i++) {
    for (size_t j = 0; j < r.size(); j++) {
      vd test_d_r(r.size(), 1.0);
      if (i > 0) {
        activity[i - 1] = 0.0;
      }
      activity[i] = 1.0;
      dd_r[i + j * r.size()] =
          __enzyme_fwddiff3(jac_fwd, enzyme_dup, r, activity, test_d_r,
                            activity2, enzyme_const, weights);
      std::cout << dd_r[i] << ", ";
    }
  }
  std::cout << "ddr" << std::endl;
}

int main(int argc, char *argv[]) {

  const vd x = {0.00000000,  0.00000000, 14.00307401, 0.00000000,
                -1.98144397, 1.00782504, -1.71598081, 0.99072198,
                1.00782504,  1.71598081, 0.99072198,  1.00782504};

  vd weights(N, 1.0);

  // Primary - correct
  double energy = f_energy(x, weights);
  std::cout << "Energy: " << energy << std::endl;

  // Jacobian - correct
  vd dx(x.size());
  vd dx1(x.size());
  jac_fwd(x, dx, weights);
  jac_rev(x, dx1, weights);
  std::cout << std::endl << "jac fwd: ";
  for (size_t i = 0; i < dx.size(); i++) {
    if (i % 3 == 0)
      std::cout << std::endl;
    std::cout << dx[i] << " \t";
  }
  std::cout << std::endl << std::endl << "jac rev: ";
  for (size_t i = 0; i < dx1.size(); i++) {
    if (i % 3 == 0)
      std::cout << std::endl;
    std::cout << dx1[i] << " \t";
  }
  std::cout << std::endl;

  // Hessian - incorrect
  vd dd_x(x.size() * x.size());
  hess_fwdfwd2(x, dd_x, weights);

  for (size_t i = 0; i < dd_x.size(); i+=3) {
    if (i % (3*4) == 0)
      std::cout << std::endl;
    if (i % (3*4*3) == 0)
      std::cout << std::endl;
    std::cout << std::setprecision(0) << std::scientific << "[" << dd_x[i] << " " <<dd_x[i+1] << " " << dd_x[i+2] << "]" << std::endl;
  }

  std::cout << std::endl;
  return 1;
}
