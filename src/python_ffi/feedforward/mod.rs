pub mod net;
pub mod sampled_trainer;
pub mod trainer;

use pyo3::{prelude::*, wrap_pymodule};

#[pymodule]
fn feedforward(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<net::Net>()?;
    m.add_class::<trainer::Trainer>()?;
    m.add_class::<sampled_trainer::SampledTrainer>()?;
    Ok(())
}

pub fn construct_module(m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(feedforward))?;
    Ok(())
}
