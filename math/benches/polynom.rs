// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand_utils::{rand_array, rand_value, rand_vector};
use std::time::Duration;
use winter_math::fields::QuadExtension;
use winter_math::{
    fft, fields::f128::BaseElement, polynom, ExtensibleField, FieldElement, StarkField,
};

const SIZES: [usize; 3] = [262_144, 524_288, 1_048_576];

fn syn_div(c: &mut Criterion) {
    let mut group = c.benchmark_group("syn_div");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES.iter() {
        let stride = 8;
        let mut values: Vec<BaseElement> = rand_vector(size);
        for v in values.iter_mut().skip(stride) {
            *v = BaseElement::ZERO;
        }
        let inv_twiddles = fft::get_inv_twiddles::<BaseElement>(size);
        fft::interpolate_poly(&mut values, &inv_twiddles);
        let p = values;
        let z_power = size / stride;

        group.bench_function(BenchmarkId::new("high_degree", size), |bench| {
            bench.iter_batched_ref(
                || p.clone(),
                |p| polynom::syn_div(p, z_power, BaseElement::ONE),
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn polynom_eval<B: StarkField + ExtensibleField<2>>(c: &mut Criterion, field_name: &str) {
    let mut group = c.benchmark_group(format!("polynom/{}", field_name));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES.iter() {
        let values: Vec<B> = rand_vector(size);
        let zs: [B; 2] = rand_array();
        let z = QuadExtension::new(zs[0], zs[1]);

        group.bench_function(BenchmarkId::new("eval", size), |bench| {
            bench.iter_batched(|| values.clone(), |p| polynom::eval(&p, z), BatchSize::SmallInput);
        });
    }

    group.finish();
}

fn eval(c: &mut Criterion) {
    polynom_eval::<winter_math::fields::f64::BaseElement>(c, "f64");
    polynom_eval::<winter_math::fields::f62::BaseElement>(c, "f62");
    polynom_eval::<winter_math::fields::f128::BaseElement>(c, "f128");
}

criterion_group!(polynom_group, eval, syn_div);
criterion_main!(polynom_group);
