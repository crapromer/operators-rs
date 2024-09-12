﻿use super::{args::Meta, Args, Rope};
use crate::{
    nvidia_gpu::{Handle as Gpu, Internal as Handle, ModuleBox},
    utils::get_or_err,
};
use common::{algebraic, locate_error, ErrorPosition, QueueOf};
use digit_layout::types::{F16, U32};
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
    max_threads_block: usize,
    scheme: Option<Arc<ModuleBox>>,
}
const NAME: &str = "rope_f16";
const CODE: &str = include_str!("rope.cuh");

impl Rope<Gpu> for Operator {}

impl common::Operator for Operator {
    type Handle = Gpu;
    type Args = Args<Gpu>;
    type SchemeError = ErrorPosition;
    type LaunchError = ErrorPosition;

    fn new(handle: &Self::Handle) -> Self {
        Self {
            handle: handle.0.clone(),
            max_threads_block: handle.0.device().block_limit().max_threads,
            scheme: None,
        }
    }

    fn scheme(&mut self, args: &Self::Args) -> Result<(), Self::SchemeError> {
        let Meta { dt_t, dt_p, .. } = args.meta()?;

        if dt_t != F16 || dt_p != U32 {
            todo!()
        }

        let cc = self.handle.device().compute_capability();
        self.scheme = Some(self.handle.compile_kernel(NAME, cc, || {
            format!(
                r#"{CODE}

extern "C" __global__ void {NAME}(
    half2 *__restrict__ t,
    int const stride_token,
    int const stride_head,
    unsigned int const *__restrict__ pos,
    float theta
){{
    padding(t, stride_token, stride_head, pos, theta);
}}"#
            )
        }));
        Ok(())
    }

    fn launch(
        &self,
        args: &Self::Args,
        queue: &QueueOf<Self::Handle>,
    ) -> Result<(), Self::LaunchError> {
        let Meta {
            dt_t, dt_p, nt, dh, ..
        } = args.meta()?;

        if dt_t != F16 || dt_p != U32 {
            todo!()
        }

        let Args {
            t_layout,
            t_base,
            p_layout,
            p_base,
            theta,
            ..
        } = args;
        let &[_, nh, _] = t_layout.shape() else {
            unreachable!()
        };
        let &[st, sh, sd] = t_layout.strides() else {
            unreachable!()
        };
        let &[sp] = p_layout.strides() else {
            unreachable!()
        };

        get_or_err!(nt);
        get_or_err!(nh);
        get_or_err!(dh);
        get_or_err!(st);
        get_or_err!(sh);
        get_or_err!(sd);
        get_or_err!(sp);

        let unit = algebraic!(dt_t)? as isize;
        if sd != unit || sp != size_of::<u32>() as isize {
            return Err(locate_error!("Unsupported layout"));
        };

        let Some(m) = self.scheme.as_ref() else {
            return Err(locate_error!("Scheme not set"));
        };

        let dh = dh / 2;
        let st = (st / unit / 2) as i32;
        let sh = (sh / unit / 2) as i32;
        let params = cuda::params![t_base, st, sh, p_base, theta];

        if self.max_threads_block % dh != 0 {
            return Err(locate_error!());
        }

        let max_nh_l = (self.max_threads_block / dh).min(nh);
        let nh_l = (1..=max_nh_l).rev().find(|nhl| nh % nhl == 0).unwrap();
        let nh_h = nh / nh_l;

        m.launch(
            CString::new(NAME).unwrap(),
            (nt as _, nh_h as _),
            (nh_l as _, dh as _),
            params.as_ptr(),
            0,
            queue,
        );
        Ok(())
    }
}

#[test]
fn test() {
    use common::{dyn_, Operator as _, TensorLayout};
    use digit_layout::types::{F32, U32};
    use std::ptr::{null, null_mut};

    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }
    let dev = cuda::Device::new(0);
    println!("{}", dev.info());

    let handle = Gpu::new(dev.context());
    let mut op = Operator::new(&handle);

    op.scheme(&Args {
        t_layout: TensorLayout::new(F16, &[dyn_(); 3], &[dyn_(); 3]),
        t_base: null_mut(),
        p_layout: TensorLayout::new(U32, &[dyn_()], &[dyn_()]),
        p_base: null(),
        sin_layout: TensorLayout::new(F32, &[dyn_(); 2], &[dyn_(); 2]),
        sin_base: null(),
        cos_layout: TensorLayout::new(F32, &[dyn_(); 2], &[dyn_(); 2]),
        cos_base: null(),
        theta: 1e4,
    })
    .unwrap();
    let module = op.scheme.as_ref().unwrap();
    handle.apply(|ctx| {
        println!(
            "{NAME}\n{}",
            module.load(CString::new(NAME).unwrap(), ctx).info()
        );
    })
}
