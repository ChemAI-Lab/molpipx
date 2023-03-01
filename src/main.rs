// fn dist(const double *a, const double *b) -> f32 {
//   float dist = std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2) +
//                std::pow(a[2] - b[2], 2);
//   return std::sqrt(dist);
// }
//

const N: usize = 51;

fn point_dist(positions: &Vec<f32>, x: usize, y: usize) -> f32 {
    let a = (positions[x] - positions[y]).powi(2);
    let b = (positions[x + 1] - positions[y + 1]).powi(2);
    let c = (positions[x + 2] - positions[y + 2]).powi(2);
    (a + b + c).sqrt()
}

fn dist(positions: &Vec<f32>) -> Vec<f32> {
    assert!(positions.len() % 3 == 0);
    let entries = positions.len() / 3;
    let mut r = vec![0.0; entries * (entries - 1) / 2];
    let mut pos = 0;

    for i in 0..entries {
        for j in (i + 1)..entries {
            r[pos] = point_dist(&positions, 3 * i, 3 * j);
            pos += 1;
        }
    }
    return r;
}

fn morse(r: Vec<f32>, l: f32) -> Vec<f32> {
    let out = r.iter().map(|x| f32::exp(-x / l)).collect();
    return out;
    // for (size_t i = 0; i < r.size(); i++) {
    //   r[i] = exp(-r[i] / l);
    // }
    // return r;
}

fn f_energy(inputs: &Vec<f32>, weights: &Vec<f32>) -> f32 {
    let distances: Vec<f32> = dist(inputs);
    let z: Vec<f32> = morse(distances, 1.0);
    let outs: Vec<f32> = vec![1.0; weights.len()];
    //let outs: Vec<f32> = f_polynomials(z);

    assert!(outs.len() == weights.len());

    let mut res = 0.0;
    for i in 0..N {
        res += weights[i] * outs[i];
    }
    return res;
}

fn main() {
    let x: Vec<f32> = vec![
        0.00000000,
        0.00000000,
        14.00307401,
        0.00000000,
        -1.98144397,
        1.00782504,
        -1.71598081,
        0.99072198,
        1.00782504,
        1.71598081,
        0.99072198,
        1.00782504,
    ];

    let weights: Vec<f32> = vec![1.0; N];

    // Primary - correct
    let energy = f_energy(&x, &weights);
    println!("Energy: {energy}");

    // Jacobian - correct
    let dx: Vec<f32> = vec![0.0; x.len()];
    let dx1: Vec<f32> = vec![0.0; x.len()];
    // jac_fwd(x, dx, weights);
    // jac_rev(x, dx1, weights);
    // std::cout << std::endl << "jac fwd: ";
    // for (size_t i = 0; i < dx.size(); i++) {
    //   if (i % 3 == 0)
    //     std::cout << std::endl;
    //   std::cout << dx[i] << " \t";
    // }
    // std::cout << std::endl << std::endl << "jac rev: ";
    // for (size_t i = 0; i < dx1.size(); i++) {
    //   if (i % 3 == 0)
    //     std::cout << std::endl;
    //   std::cout << dx1[i] << " \t";
    // }
    // std::cout << std::endl;

    // // Hessian - incorrect
    // vd dd_x(x.size() * x.size());
    // hess_fwdfwd2(x, dd_x, weights);

    // for (size_t i = 0; i < dd_x.size(); i++) {
    //   if (i % x.size() == 0)
    //     std::cout << std::endl;
    //   std::cout << "\t" << i << ": \t" << dd_x[i];
    // }
}
