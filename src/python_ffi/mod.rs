//! Python FFI
use pyo3::prelude::*;

pub mod feedforward;
mod gen_macros;

#[pymodule]
fn rpnnet(_py: Python, m: &PyModule) -> PyResult<()> {
    feedforward::construct_module(m)?;
    Ok(())
}
