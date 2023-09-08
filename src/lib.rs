#![feature(generic_const_exprs)]

fn point_dist<const NR: usize>(positions: &[f32; NR], x: usize, y: usize) -> f32 {
    let a = (positions[x] - positions[y]).powi(2);
    let b = (positions[x + 1] - positions[y + 1]).powi(2);
    let c = (positions[x + 2] - positions[y + 2]).powi(2);
    (a + b + c).sqrt()
}

pub const fn get_len(x: usize) -> usize {
    assert!(x % 3 == 0);
    let n = x / 3;
    (n * (n - 1)) / 2
}

pub fn dist<const NR: usize>(positions: &[f32; NR]) -> [f32; get_len(NR)] {
    assert!(NR % 3 == 0);
    let mut r = [0.0; get_len(NR)];
    let mut pos = 0;

    for i in 0..(NR / 3) {
        for j in (i + 1)..(NR / 3) {
            r[pos] = point_dist::<NR>(positions, 3 * i, 3 * j);
            pos += 1;
        }
    }
    return r;
}

pub fn morse(r: &mut [f32], l: f32) {
    for x in r {
        *x = f32::exp(-*x / l);
    }
    //r = r.iter().map(|x| f32::exp(-x / l)).collect();
}

