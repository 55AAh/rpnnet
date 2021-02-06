use pyo3::prelude::*;

use super::{
    net::{ConsumableNet, Net},
    sampled_trainer::{ConsumableSampledTrainer, SampledTrainer},
};
use crate::feedforward::{ProcessError, TrainError, Trainer as InnerTrainer};
use crate::{Impl_to_PyErr, MakeConsumable};

MakeConsumable!(ConsumableTrainer, InnerTrainer, Trainer);

#[pyclass]
pub struct Trainer {
    pub(super) trainer: ConsumableTrainer,
    pub(super) outputs_buffer: Box<[f64]>,
}

#[pymethods]
impl Trainer {
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

    pub fn train(
        &mut self,
        inputs: Vec<f64>,
        desired_outputs: Vec<f64>,
        grad_mult_coeff: f64,
    ) -> Result<f64, TrainError> {
        self.trainer.get_ref_mut().train(
            &inputs,
            &mut self.outputs_buffer,
            &desired_outputs,
            grad_mult_coeff,
        )
    }

    pub fn apply_training(&mut self) {
        self.trainer.get_ref_mut().apply_training();
    }

    #[staticmethod]
    pub fn _flatten_training_data(
        samples: Vec<(Vec<f64>, Vec<f64>)>,
        input_layer_size: usize,
        output_layer_size: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), TrainError> {
        let samples_count = samples.len();

        let mut all_inputs_vec = Vec::with_capacity(samples_count * input_layer_size);
        let mut all_desired_outputs_vec = Vec::with_capacity(samples_count * output_layer_size);

        for (i, (inputs, desired_outputs)) in samples.into_iter().enumerate() {
            if inputs.len() != input_layer_size {
                return Err(TrainError::WrongSampleInputsCount((
                    i,
                    input_layer_size,
                    inputs.len(),
                )));
            }
            all_inputs_vec.extend(inputs);

            if desired_outputs.len() != output_layer_size {
                return Err(TrainError::WrongSampleDesiredOutputsCount((
                    i,
                    output_layer_size,
                    desired_outputs.len(),
                )));
            }
            all_desired_outputs_vec.extend(desired_outputs);
        }

        Ok((all_inputs_vec, all_desired_outputs_vec))
    }

    pub fn train_batch(
        &mut self,
        samples: Vec<(Vec<f64>, Vec<f64>)>,
        grad_mult_coeff: f64,
        get_outputs: bool,
    ) -> Result<(f64, Option<Vec<Vec<f64>>>), TrainError> {
        let samples_len = samples.len();
        let geometry = self.trainer.get_ref().net_ref().geometry();
        let input_layer_size = geometry[0];
        let output_layer_size = *geometry.last().unwrap();
        let (all_inputs_vec, all_desired_outputs_vec) =
            Trainer::_flatten_training_data(samples, input_layer_size, output_layer_size)?;

        let all_outputs_count = if get_outputs {
            all_desired_outputs_vec.len()
        } else {
            output_layer_size
        };
        let mut all_outputs_vec = Vec::with_capacity(all_outputs_count);
        all_outputs_vec.extend((0..all_outputs_count).map(|_| 0.0));

        Ok((
            self.trainer.get_ref_mut().train_batch(
                samples_len,
                &all_inputs_vec,
                &mut all_outputs_vec,
                get_outputs,
                &all_desired_outputs_vec,
                grad_mult_coeff,
            )?,
            if get_outputs {
                Some(
                    all_outputs_vec
                        .chunks(output_layer_size)
                        .map(|o| o.to_owned())
                        .collect(),
                )
            } else {
                None
            },
        ))
    }

    pub fn train_random(
        &mut self,
        samples_count: usize,
        batch_size: usize,
        samples: Vec<(Vec<f64>, Vec<f64>)>,
        grad_mult_coeff: f64,
        get_outputs: bool,
    ) -> Result<(f64, Option<(Vec<usize>, Vec<Vec<f64>>)>), TrainError> {
        let selection_size = samples.len();
        if batch_size < 1 || batch_size > samples_count {
            return Err(TrainError::BadBatchSize((samples_count, batch_size)));
        }

        let geometry = self.trainer.get_ref().net_ref().geometry();
        let input_layer_size = geometry[0];
        let output_layer_size = *geometry.last().unwrap();
        let (all_inputs_vec, all_desired_outputs_vec) =
            Trainer::_flatten_training_data(samples, input_layer_size, output_layer_size)?;

        if get_outputs {
            let mut inputs_indices = Vec::with_capacity(samples_count);
            inputs_indices.extend((0..samples_count).map(|_| 0));

            let all_outputs_count = output_layer_size * samples_count;
            let mut outputs = Vec::with_capacity(all_outputs_count);
            outputs.extend((0..all_outputs_count).map(|_| 0.0));

            Ok((
                self.trainer.get_ref_mut().train_random(
                    selection_size,
                    samples_count,
                    batch_size,
                    &all_inputs_vec,
                    Some(inputs_indices.as_mut()),
                    &mut outputs,
                    get_outputs,
                    &all_desired_outputs_vec,
                    grad_mult_coeff,
                )?,
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
                    selection_size,
                    samples_count,
                    batch_size,
                    &all_inputs_vec,
                    None,
                    &mut self.outputs_buffer,
                    get_outputs,
                    &all_desired_outputs_vec,
                    grad_mult_coeff,
                )?,
                None,
            ))
        }
    }

    pub fn build_sampled(
        &mut self,
        samples: Vec<(Vec<f64>, Vec<f64>)>,
    ) -> Result<SampledTrainer, TrainError> {
        let samples_count = samples.len();
        let geometry = self.trainer.get_ref().net_ref().geometry();
        let input_layer_size = geometry[0];
        let output_layer_size = *geometry.last().unwrap();
        let (all_inputs_vec, all_desired_outputs_vec) =
            Trainer::_flatten_training_data(samples, input_layer_size, output_layer_size)?;

        Ok(SampledTrainer {
            trainer: ConsumableSampledTrainer::acquire(self.trainer.release()),
            selection_size: samples_count,
            all_inputs: all_inputs_vec.into_boxed_slice(),
            outputs_buffer: self.outputs_buffer.clone(),
            all_desired_outputs: all_desired_outputs_vec.into_boxed_slice(),
        })
    }

    pub fn teardown(&mut self) -> Net {
        Net {
            net: ConsumableNet::acquire(self.trainer.release().net),
            outputs_buffer: self.outputs_buffer.clone(),
        }
    }
}

Impl_to_PyErr!(for TrainError);
