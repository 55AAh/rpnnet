use pyo3::prelude::*;

use super::trainer::{ConsumableTrainer, Trainer};
use crate::feedforward::{Net as InnerNet, NewNetError, ProcessError, SizeMismatch};
use crate::{Impl_to_PyErr, MakeConsumable};

MakeConsumable!(ConsumableNet, InnerNet, Net);

#[pyclass]
pub struct Net {
    pub(super) net: ConsumableNet,
    pub(super) outputs_buffer: Box<[f64]>,
}

#[pymethods]
impl Net {
    #[new]
    pub fn new(geometry: Vec<usize>, coefficients: Option<Vec<f64>>) -> Result<Self, NewNetError> {
        let net = InnerNet::new(
            &geometry,
            coefficients.and_then(|v| Some(v.into_boxed_slice())),
        )?;

        let outputs_count = *geometry.last().unwrap();
        let mut outputs_buffer = Vec::with_capacity(outputs_count);
        outputs_buffer.extend((0..outputs_count).map(|_| 0.0));

        Ok(Self {
            net: ConsumableNet::acquire(net),
            outputs_buffer: outputs_buffer.into_boxed_slice(),
        })
    }

    pub fn geometry(&self) -> Vec<usize> {
        self.net.get_ref().geometry().to_vec()
    }

    pub fn export(&self) -> (Vec<usize>, Vec<f64>) {
        let (geometry, coeffs) = self.net.get_ref().export();
        (geometry.to_owned(), coeffs.to_owned())
    }

    pub fn process(&mut self, inputs: Vec<f64>) -> Result<Vec<f64>, ProcessError> {
        self.net
            .get_ref_mut()
            .process(inputs.as_ref(), &mut self.outputs_buffer)?;
        Ok(self.outputs_buffer.to_vec())
    }

    pub fn build_trainer(&mut self) -> Trainer {
        Trainer {
            trainer: ConsumableTrainer::acquire(self.net.release().build_trainer()),
            outputs_buffer: self.outputs_buffer.clone(),
        }
    }

    #[staticmethod]
    pub fn calc_cost(outputs: Vec<f64>, desired_outputs: Vec<f64>) -> Result<f64, SizeMismatch> {
        Ok(InnerNet::cals_cost(&outputs, &desired_outputs)?)
    }
}

Impl_to_PyErr!(for NewNetError, ProcessError, SizeMismatch);
