use rand::prelude::Distribution;
use std::{fmt, mem};

use super::net::{Net, SizeMismatch};

/// Net trainer structure.
///
/// To train Net, additional buffers are needed. We will contain them in this structure.
/// Training procedure will look like this:
/// * One allocates additional buffers by calling `Net::build_trainer`, which will consume `Net`
/// and return `Trainer` object.
/// (Consuming `Net` is needed to prevent one from building another concurrent `Trainer`s.)
/// * Training data is processed via `Trainer::train` and `Trainer::apply_training` methods.
/// Also, at any time one can call `Trainer::net` to get access to `Net::process`.
/// * Cost estimation is possible via `Net::process` on a test inputs, folowed by
/// `Net::calc_cost` on received and desired outputs. Additionally, when training,
/// ouput values and costs will be returned by all methods, see their documentation.
/// * Once finished training, one can use `Trainer::free` to free all the additional buffers,
/// allocated in Trainer, and get `Net` object back.
pub struct Trainer {
    /// The network object trainer posesses.  
    /// (Its buffer will be used for activations derivatives in the same manner as it's used
    /// for the activations in `Net::process`).
    pub(crate) net: Net,

    /// Since we will need to store activations of all layers for backpropagation, this is
    /// the buffer that will hold them:
    /// `buffer = [layer_1][layer_2] ... [layer_N]
    pub(crate) buffer: Box<[f64]>,

    /// Buffer with equivalent structure and size as `Net::coeffs`, that will hold it's
    /// gradient changes
    pub(crate) gradient_buffer: Box<[f64]>,

    // Samples counter for mean gradient
    pub(crate) gradient_counter: usize,
}

impl Trainer {
    /// Consumes `Net` and builds `Trainer` object containing it.
    pub(super) fn build(net: Net) -> Trainer {
        // Buffer for activations will hold all layers, except 0 (input)
        let buffer_size = net.geometry[1..net.geometry.len() - 1].iter().sum();
        let mut buffer = Vec::with_capacity(buffer_size);
        buffer.extend((0..buffer_size).map(|_| 0.0));

        // Gradient buffer contains changes to Net::coeffs, so it is the same size
        let mut gradient_buffer: Vec<f64> = Vec::with_capacity(net.coeffs.len());
        gradient_buffer.extend((0..net.coeffs.len()).map(|_| 0.0));

        Trainer {
            net,
            buffer: buffer.into_boxed_slice(),
            gradient_buffer: gradient_buffer.into_boxed_slice(),
            gradient_counter: 0,
        }
    }

    /// Returns reference to contained `Net`, allowing the use of `Net::process`.
    pub fn net_ref(&self) -> &Net {
        &self.net
    }

    /// Returns mutable reference to contained `Net`, allowing the use of `Net::process`.
    pub fn net_mut(&mut self) -> &mut Net {
        &mut self.net
    }

    /// Used in Trainer::process_der_iteration to update neuron gradient buffer.
    ///
    /// # Arguments
    /// * `coeff` - multiplier coefficient;
    /// * `gradbuf` - gradient buffer;
    /// * `new_activations` - slice that holds activations of new layer.
    fn update_gradbuf(coeff: f64, gradbuf: &mut [f64], new_activations: &[f64]) {
        for (gc, &na) in gradbuf[..new_activations.len()]
            .iter_mut()
            .zip(new_activations.iter())
        {
            *gc += coeff * na;
        }
        gradbuf[new_activations.len()] += coeff;
    }

    /// Used in Trainer::train for derivatives computing iterations.  
    ///
    /// # Arguments
    /// * `old_layer_size` - mutable reference to `old_layer_size`;
    /// * `current_layer_size` - mutable reference to `current_layer_size`;
    /// * `new_layer_size` - new layer size;
    /// * `remaining_coeffs` - mutable remaining coefficients slice;
    /// * `remaining_activations` - mutable remaining activations slice;
    /// * `current_activations` - mutable current activations slice;
    /// * `old_activations` - mutable to old activations slice;
    /// * `old_buffer` - slice of old derivatives buffer;
    /// * `buffer` - slice of current derivatives mutable buffer;
    /// * `remaining_gradbuf` - mutable `remaining_gradbuf` slice;
    /// * `grad_mult_coeff` - gradient multiplier.
    fn process_der_iteration<'a>(
        old_layer_size: &mut usize,
        current_layer_size: &mut usize,
        new_layer_size: usize,
        remaining_coeffs: &mut &[f64],
        remaining_activations: &mut &'a [f64],
        current_activations: &mut &'a [f64],
        old_activations: &mut &'a [f64],
        old_buffer: &[f64],
        buffer: &mut [f64],
        remaining_gradbuf: &mut &mut [f64],
        grad_mult_coeff: f64,
    ) {
        // We need to use coeffs of old layer, it has old_layer_size neurons,
        // and each of them has current_layer_size weights + bias.
        // Since we're going in reverse order, we'll cut corresponding piece from the right
        let (rem_coeffs, old_coeffs) = (*remaining_coeffs)
            .split_at((*remaining_coeffs).len() - (*old_layer_size * (*current_layer_size + 1)));
        *remaining_coeffs = rem_coeffs;

        // We need to use activations of old, current and new layers, so we have to iterate
        // with "[remaining][new][current][old]".
        // Since we're going in reverse order, we'll cut corresponding piece from the right
        let (rem_activations, new_activations) =
            (*remaining_activations).split_at((*remaining_activations).len() - new_layer_size);
        *remaining_activations = rem_activations;

        // We need to use gradbuf of current layer, it has current_layer_size neurons,
        // and each of them has new_layer_size weights + bias.
        // Since we're going in reverse order, we'll cut corresponding piece from the right
        let (rem_gradbuf, mut gradbuf_curr_rem) = mem::take(remaining_gradbuf)
            .split_at_mut((*remaining_coeffs).len() - (*current_layer_size * (new_layer_size + 1)));
        *remaining_gradbuf = rem_gradbuf;

        // For each neuron in layer
        for (i, &cs) in (0..*current_layer_size).zip((*current_activations).iter()) {
            // Getting slice of current gradbuf (weights + bias)
            let (current_gradbuf, gradbuf_tail) = gradbuf_curr_rem.split_at_mut(new_layer_size + 1);
            // And advancing forward
            gradbuf_curr_rem = gradbuf_tail;

            // Calculating and storing derivative for this neuron
            let der = old_buffer[..*old_layer_size]
                .iter()
                .zip(old_activations.iter())
                .zip((0..*old_layer_size).map(|j| old_coeffs[(*current_layer_size + 1) * j + i]))
                .map(|((&d, &s), c)| d * Net::sigmoid_der_s(s) * c)
                .sum();
            buffer[i] = der;

            // Updating gradbuf
            let der_coeff = der * Net::sigmoid_der_s(cs) * grad_mult_coeff;
            Trainer::update_gradbuf(der_coeff, current_gradbuf, new_activations);
        }

        *old_activations = current_activations;
        *current_activations = new_activations;
        *old_layer_size = *current_layer_size;
        *current_layer_size = new_layer_size;
    }

    /// Performs training process on a given sample, updating `Trainer`'s internal gradient buffer.
    /// Note that training updates are not done by this method, allowing one to use stochastic
    /// gradient descent over many samples. (`Trainer::train_batch` does this.)
    /// To finally aplly changes, call `Trainer::apply_training`.
    ///
    /// # Arguments
    /// * `inputs` - slice that holds activations of input neurons;
    /// * `outputs` - mutable slice that will be filled with activations of output neurons;
    /// * `desired_outputs` - slice that holds desired activations of outputs neurons;
    /// * `grad_mult_coef` - gradient multiplier.
    ///
    /// # Returns
    /// * The cost function of resulting `outputs` and given `desired_outputs`.
    ///
    /// # Examples
    /// ```
    /// # use rpnnet::feedforward::Net;
    /// let mut trainer = Net::new(&[10, 20, 20, 3], None).unwrap().build_trainer();
    /// let inputs = [1.0; 10];
    /// let desired_outputs = [2.0; 3];
    /// let mut outputs = [0.0, 0.0, 0.0];
    /// let cost = trainer.train(&inputs, &mut outputs, &desired_outputs, 0.001).unwrap();
    /// ```
    pub fn train(
        &mut self,
        inputs: &[f64],
        outputs: &mut [f64],
        desired_outputs: &[f64],
        grad_mult_coeff: f64,
    ) -> Result<f64, TrainError> {
        let layers_count = self.net.geometry.len();

        if inputs.len() != self.net.geometry[0] {
            return Err(TrainError::BadInputs(SizeMismatch {
                expected: self.net.geometry[0],
                got: inputs.len(),
            }));
        }
        if outputs.len() != self.net.geometry[layers_count - 1] {
            return Err(TrainError::BadOutputs(SizeMismatch {
                expected: self.net.geometry[layers_count - 1],
                got: outputs.len(),
            }));
        }
        if desired_outputs.len() != outputs.len() {
            return Err(TrainError::BadDesiredOutputs(SizeMismatch {
                expected: outputs.len(),
                got: desired_outputs.len(),
            }));
        }

        // Forward activations calculation
        let mut remaining_coeffs = self.net.coeffs.as_ref();
        let (mut old_layer_size, layer_size) = (self.net.geometry[0], self.net.geometry[1]);

        if layers_count == 2 {
            // The only iteration: inputs -> outputs
            Net::process_act_iteration(
                &mut old_layer_size,
                layer_size,
                &mut remaining_coeffs,
                inputs,
                outputs,
            );
        } else {
            let mut buffer = self.buffer.as_mut();

            // First iteration: inputs -> old_buffer
            Net::process_act_iteration(
                &mut old_layer_size,
                layer_size,
                &mut remaining_coeffs,
                inputs,
                buffer,
            );

            // Intermediate iterations: old_buffer -> buffer, then advance buffers
            for &layer_size in self.net.geometry[2..layers_count - 1].iter() {
                let (old_buffer, _buffer) = buffer.split_at_mut(old_layer_size);
                buffer = _buffer;

                Net::process_act_iteration(
                    &mut old_layer_size,
                    layer_size,
                    &mut remaining_coeffs,
                    old_buffer,
                    buffer,
                );
            }

            // Last iteration: buffer -> outputs
            Net::process_act_iteration(
                &mut old_layer_size,
                self.net.geometry[layers_count - 1],
                &mut remaining_coeffs,
                buffer,
                outputs,
            );
        }

        // Backpropagation
        let mut old_layer_size = self.net.geometry[layers_count - 1];
        let mut current_layer_size = self.net.geometry[layers_count - 2];
        let mut remaining_coeffs = self.net.coeffs.as_ref();
        let (mut old_buffer, mut buffer) = self.net.buffer.split_at_mut(self.net.buffer.len() / 2);

        let mut cost = 0.0;
        // Derivatives of output layer: outputs -> old_buffer
        for ((&o, &d_o), b) in outputs
            .iter()
            .zip(desired_outputs.iter())
            .zip(old_buffer.iter_mut())
        {
            let diff = o - d_o;
            *b = 2.0 * diff;
            cost += diff * diff;
        }

        // We need to update gradbuf of output layer, it has layer_N_size neurons,
        // and each of them has layer_N-1_size weights + bias.
        // Since we're going in reverse order, we'll cut corresponding piece from the right
        let (mut remaining_gradbuf, mut gradbuf_curr_rem) = self
            .gradient_buffer
            .split_at_mut(self.gradient_buffer.len() - (old_layer_size * (current_layer_size + 1)));

        if layers_count == 2 {
            // For each neuron in output layer
            for (&cs, &der) in outputs.iter().zip(old_buffer.iter()) {
                // Getting slice of current gradbuf (weights + bias)
                let (current_gradbuf, gradbuf_tail) =
                    gradbuf_curr_rem.split_at_mut(current_layer_size + 1);
                // And advancing forward
                gradbuf_curr_rem = gradbuf_tail;

                // Updating gradbuf of input layer
                let der_coeff = der * Net::sigmoid_der_s(cs) * grad_mult_coeff;
                Trainer::update_gradbuf(der_coeff, current_gradbuf, inputs);
            }
        } else {
            let (mut remaining_activations, mut current_activations) = self
                .buffer
                .as_ref()
                .split_at(self.buffer.len() - current_layer_size);

            // For each layer in reverse (except last and first two)
            let mut old_activations = &*outputs;
            for &new_layer_size in self.net.geometry.iter().skip(1).rev().skip(2) {
                Trainer::process_der_iteration(
                    &mut old_layer_size,
                    &mut current_layer_size,
                    new_layer_size,
                    &mut remaining_coeffs,
                    &mut remaining_activations,
                    &mut current_activations,
                    &mut old_activations,
                    old_buffer,
                    buffer,
                    &mut remaining_gradbuf,
                    grad_mult_coeff,
                );
                mem::swap(&mut buffer, &mut old_buffer);
            }

            // For second layer (it's remaing_activations are inputs)
            remaining_activations = inputs;
            Trainer::process_der_iteration(
                &mut old_layer_size,
                &mut current_layer_size,
                self.net.geometry[0],
                &mut remaining_coeffs,
                &mut remaining_activations,
                &mut current_activations,
                &mut old_activations,
                old_buffer,
                buffer,
                &mut remaining_gradbuf,
                grad_mult_coeff,
            )
        }

        self.gradient_counter += 1;

        Ok(cost)
    }

    /// Applies training, previously done by `Trainer::train`.
    pub fn apply_training(&mut self) {
        if self.gradient_counter > 0 {
            for (c, g) in self
                .net
                .coeffs
                .iter_mut()
                .zip(self.gradient_buffer.iter_mut())
            {
                *c -= *g / self.gradient_counter as f64;
                *g = 0.0;
            }
            self.gradient_counter = 0;
        }
    }

    /// Performs training for every data sample in a given batch, then applies.  
    /// (It is equivalent to calling `Trainer::train` repeatedly, and then `Trainer:apply_training`.)  
    /// This allows one to use stochastic gradient descent.
    ///
    /// # Arguments
    /// * `batch_size` - number of samples in batch;
    /// * `all_inputs` - slice of all samples' inputs, joined;
    /// * `outputs` - mutable slice for activations of output neurons;
    /// * `outputs_all` - boolean, determines whether `outputs`:
    ///   * `true` - should be filled with all activations, joined, and have size of `desired outputs`;
    ///   * `false` - should be used as buffer and have size of last layer.
    /// * `all_desired_outputs` - slice of all samples' desired_outputs, joined;
    /// * `grad_mult_coef` - gradient multiplier.
    ///
    /// # Returns
    /// * The average value of cost function of resulting `outputs` and given `desired_outputs`.
    ///
    /// # Examples
    /// ```
    /// # use rpnnet::feedforward::Net;
    /// let mut trainer = Net::new(&[2, 10, 10, 1], None).unwrap().build_trainer();
    /// let inputs = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    /// let desired_outputs = [0.0, -1.0, -2.0];
    ///
    /// // Using outputs as buffer
    /// let mut outputs = [0.0];
    /// let cost = trainer.train_batch(3, &inputs, &mut outputs, false, &desired_outputs, 0.001).unwrap();
    ///
    /// // Using outputs as samples result
    /// let mut outputs = [0.0, 0.0, 0.0];
    /// let cost = trainer.train_batch(3, &inputs, &mut outputs, true, &desired_outputs, 0.001).unwrap();
    /// ```
    pub fn train_batch(
        &mut self,
        batch_size: usize,
        all_inputs: &[f64],
        outputs: &mut [f64],
        outputs_all: bool,
        all_desired_outputs: &[f64],
        grad_mult_coeff: f64,
    ) -> Result<f64, TrainError> {
        let inputs_len = self.net.geometry[0];
        let outputs_len = *self.net.geometry.last().unwrap();

        if all_inputs.len() != batch_size * inputs_len {
            return Err(TrainError::BadInputs(SizeMismatch {
                expected: batch_size * inputs_len,
                got: all_inputs.len(),
            }));
        }
        if all_desired_outputs.len() != batch_size * outputs_len {
            return Err(TrainError::BadDesiredOutputs(SizeMismatch {
                expected: batch_size * outputs_len,
                got: all_desired_outputs.len(),
            }));
        }
        if outputs_all {
            if outputs.len() != batch_size * outputs_len {
                return Err(TrainError::BadOutputs(SizeMismatch {
                    expected: batch_size * outputs_len,
                    got: outputs.len(),
                }));
            }
        } else {
            if outputs.len() != outputs_len {
                return Err(TrainError::BadOutputs(SizeMismatch {
                    expected: outputs_len,
                    got: outputs.len(),
                }));
            }
        }

        if batch_size == 0 {
            return Ok(0.0);
        }

        let mut costs_sum = 0.0;

        if outputs_all {
            for ((inputs, desired_outputs), outputs) in all_inputs
                .chunks(inputs_len)
                .zip(all_desired_outputs.chunks(outputs_len))
                .zip(outputs.chunks_mut(outputs_len))
            {
                costs_sum += self
                    .train(inputs, outputs, desired_outputs, grad_mult_coeff)
                    .unwrap();
            }
        } else {
            for (inputs, desired_outputs) in all_inputs
                .chunks(inputs_len)
                .zip(all_desired_outputs.chunks(outputs_len))
            {
                costs_sum += self
                    .train(inputs, outputs, desired_outputs, grad_mult_coeff)
                    .unwrap();
            }
        }

        self.apply_training();

        Ok(costs_sum / batch_size as f64)
    }

    /// Performs training for `samples_count` uniformly selected data samples, then applies.  
    /// (It is equivalent to calling `Trainer::train` repeatedly, and then `Trainer:apply_training`.)  
    /// This allows one to use stochastic gradient descent.
    ///
    /// # Arguments
    /// * `selection_size` - number of samples in selection;
    /// * `samples_count` - number of training iterations;
    /// * `batch_count` - number of samples in batch;
    /// * `all_inputs` - slice of all samples' inputs, joined;
    /// * `inputs_indices` - mutable slice for selected indeces, should be empty if `outputs_all` = false;
    /// * `outputs` - mutable slice for activations of output neurons;
    /// * `outputs_all` - boolean, determines whether `outputs`:
    ///   * `true` - should be filled with all activations, joined, and have length
    /// `size(last_layer_size) * samples_count`;
    ///   * `false` - should be used as buffer and have size of last layer.
    /// * `all_desired_outputs` - slice of all samples' desired_outputs, joined;
    /// * `grad_mult_coef` - gradient multiplier.
    ///
    /// # Returns
    /// * The average value of cost function of resulting `outputs` and given `desired_outputs`.
    ///
    /// # Examples
    /// ```
    /// # use rpnnet::feedforward::Net;
    /// let mut trainer = Net::new(&[2, 10, 10, 1], None).unwrap().build_trainer();
    /// let inputs = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    /// let desired_outputs = [0.0, -1.0, -2.0];
    ///
    /// // Using outputs as buffer
    /// let mut outputs = [0.0];
    /// let cost = trainer.train_random(3, 10, 5, &inputs, Some(vec![].as_mut()),
    ///                                 &mut outputs, false, &desired_outputs, 0.001).unwrap();
    ///
    /// // Using outputs as samples result
    /// let mut outputs = [0.0; 100];
    /// let mut indeces = [0; 100];
    /// let cost = trainer.train_random(3, 100, 10, &inputs, Some(indeces.as_mut()),
    ///                                 &mut outputs, true, &desired_outputs, 0.001).unwrap();
    /// ```
    pub fn train_random(
        &mut self,
        selection_size: usize,
        samples_count: usize,
        batch_size: usize,
        all_inputs: &[f64],
        inputs_indices: Option<&mut [usize]>,
        outputs: &mut [f64],
        outputs_all: bool,
        all_desired_outputs: &[f64],
        grad_mult_coeff: f64,
    ) -> Result<f64, TrainError> {
        let inputs_len = self.net.geometry[0];
        let outputs_len = *self.net_ref().geometry.last().unwrap();

        if batch_size == 0 || batch_size > samples_count {
            return Err(TrainError::BadBatchSize((samples_count, batch_size)));
        }

        if all_inputs.len() != selection_size * inputs_len {
            return Err(TrainError::BadInputs(SizeMismatch {
                expected: selection_size * inputs_len,
                got: all_inputs.len(),
            }));
        }
        if all_desired_outputs.len() != selection_size * outputs_len {
            return Err(TrainError::BadDesiredOutputs(SizeMismatch {
                expected: selection_size * outputs_len,
                got: all_desired_outputs.len(),
            }));
        }
        if outputs_all {
            if outputs.len() != samples_count * outputs_len {
                return Err(TrainError::BadOutputs(SizeMismatch {
                    expected: samples_count * outputs_len,
                    got: outputs.len(),
                }));
            }
            if inputs_indices.as_ref().unwrap().len() != samples_count {
                return Err(TrainError::BadIndices(SizeMismatch {
                    expected: samples_count,
                    got: inputs_indices.unwrap().len(),
                }));
            }
        } else {
            if outputs.len() != outputs_len {
                return Err(TrainError::BadOutputs(SizeMismatch {
                    expected: outputs_len,
                    got: outputs.len(),
                }));
            }
        }

        if selection_size == 0 || samples_count == 0 {
            return Ok(0.0);
        }

        let mut costs_sum = 0.0;
        let mut rng = rand::thread_rng();
        let indeces_between = rand::distributions::Uniform::from(0..selection_size);

        if outputs_all {
            for (i, (index, outputs)) in inputs_indices
                .unwrap()
                .iter_mut()
                .zip(outputs.chunks_mut(outputs_len))
                .enumerate()
            {
                if i % batch_size == 0 {
                    self.apply_training();
                }
                let random_index = indeces_between.sample(&mut rng);
                *index = random_index;
                let inputs =
                    &all_inputs[random_index * inputs_len..(random_index + 1) * inputs_len];
                let desired_outputs = &all_desired_outputs
                    [random_index * outputs_len..(random_index + 1) * outputs_len];
                costs_sum += self
                    .train(inputs, outputs, desired_outputs, grad_mult_coeff)
                    .unwrap();
            }
        } else {
            for i in 0..samples_count {
                if i % batch_size == 0 {
                    self.apply_training();
                }
                let random_index = indeces_between.sample(&mut rng);
                let inputs =
                    &all_inputs[random_index * inputs_len..(random_index + 1) * inputs_len];
                let desired_outputs = &all_desired_outputs
                    [random_index * outputs_len..(random_index + 1) * outputs_len];
                costs_sum += self
                    .train(inputs, outputs, desired_outputs, grad_mult_coeff)
                    .unwrap();
            }
        }

        self.apply_training();

        Ok(costs_sum / samples_count as f64)
    }

    /// Frees training buffers, consuming `Trainer` object, and returns contained `Net` back.  
    /// Note that all unapplied training, done by `Trainer::train` will be lost,
    /// so don't forget to call `Trainer::aplly_training` before!
    pub fn teardown(self) -> Net {
        self.net
    }
}

#[derive(Debug, Clone)]
pub enum TrainError {
    BadInputs(SizeMismatch),
    WrongSampleInputsCount((usize, usize, usize)),
    BadOutputs(SizeMismatch),
    BadDesiredOutputs(SizeMismatch),
    WrongSampleDesiredOutputsCount((usize, usize, usize)),
    BadBatchSize((usize, usize)),
    BadIndices(SizeMismatch),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            TrainError::BadInputs(SizeMismatch { expected, got }) => {
                write!(f, "Expected {} input(s), but got {}!", expected, got)
            }
            TrainError::WrongSampleInputsCount((i, expected, got)) => write!(
                f,
                "Expected {} input(s), but samples[{}] got {}!",
                expected, i, got
            ),
            TrainError::BadOutputs(SizeMismatch { expected, got }) => {
                write!(f, "Expected {} output(s), but got {}!", expected, got)
            }
            TrainError::BadDesiredOutputs(SizeMismatch { expected, got }) => write!(
                f,
                "Expected {} desired output(s), but got {}!",
                expected, got
            ),
            TrainError::WrongSampleDesiredOutputsCount((i, expected, got)) => write!(
                f,
                "Expected {} desired output(s), but samples[{}] got {}!",
                expected, i, got
            ),
            TrainError::BadBatchSize((samples_count, batch_size)) => write!(
                f,
                "Batch size must be from 1 to {}, but got {}!",
                samples_count, batch_size
            ),
            TrainError::BadIndices(SizeMismatch { expected, got }) => write!(
                f,
                "Expected {} indices output(s), but got {}!",
                expected, got
            ),
        }
    }
}
