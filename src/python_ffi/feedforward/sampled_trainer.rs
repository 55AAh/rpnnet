use pyo3::prelude::*;

use super::trainer::{ConsumableTrainer, Trainer};
use crate::feedforward::{ProcessError, TrainError, Trainer as InnerTrainer};
use crate::MakeConsumable;

MakeConsumable!(ConsumableSampledTrainer, InnerTrainer, SampledTrainer);

#[pyclass]
pub struct SampledTrainer {
    pub(super) trainer: ConsumableSampledTrainer,
    pub(super) selection_size: usize,
    pub(super) all_inputs: Box<[f64]>,
    pub(super) outputs_buffer: Box<[f64]>,
    pub(super) all_desired_outputs: Box<[f64]>,
}

#[pymethods]
impl SampledTrainer {
    pub fn geometry(&self) -> Vec<usize> {
        self.trainer.get_ref().net_ref().geometry().to_vec()
    }

    pub fn export_net(&self) -> (Vec<usize>, Vec<f64>) {
        let (geometry, coeffs) = self.trainer.get_ref().net_ref().export();
        (geometry.to_owned(), coeffs.to_owned())
    }

    pub fn process(&mut self, inputs: Vec<f64>) -> Result<Vec<f64>, ProcessError> {
        self.trainer
            .get_ref_mut()
            .net_mut()
            .process(&inputs, &mut self.outputs_buffer)?;
        Ok(self.outputs_buffer.to_vec())
    }

    pub fn train_random(
        &mut self,
        samples_count: usize,
        batch_size: usize,
        grad_mult_coeff: f64,
        get_outputs: bool,
    ) -> Result<(f64, Option<(Vec<usize>, Vec<Vec<f64>>)>), TrainError> {
        if get_outputs {
            let geometry = self.trainer.get_ref().net_ref().geometry();
            let output_layer_size = *geometry.last().unwrap();

            let mut inputs_indices = Vec::with_capacity(samples_count);
            inputs_indices.extend((0..samples_count).map(|_| 0));

            let all_outputs_count = output_layer_size * samples_count;
            let mut outputs = Vec::with_capacity(all_outputs_count);
            outputs.extend((0..all_outputs_count).map(|_| 0.0));

            let cost = self.trainer.get_ref_mut().train_random(
                self.selection_size,
                samples_count,
                batch_size,
                &self.all_inputs,
                Some(&mut inputs_indices),
                &mut outputs,
                true,
                &self.all_desired_outputs,
                grad_mult_coeff,
            )?;

            Ok((
                cost,
                Some((
                    inputs_indices,
                    outputs
                        .chunks(output_layer_size)
                        .map(|o| o.to_owned())
                        .collect(),
                )),
            ))
        } else {
            Ok((
                self.trainer.get_ref_mut().train_random(
                    self.selection_size,
                    samples_count,
                    batch_size,
                    &self.all_inputs,
                    None,
                    &mut self.outputs_buffer,
                    false,
                    &self.all_desired_outputs,
                    grad_mult_coeff,
                )?,
                None,
            ))
        }
    }

    pub fn teardown(&mut self) -> Trainer {
        Trainer {
            trainer: ConsumableTrainer::acquire(self.trainer.release()),
            outputs_buffer: self.outputs_buffer.clone(),
        }
    }
}
