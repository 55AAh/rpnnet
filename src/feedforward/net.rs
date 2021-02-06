use rand::prelude::Distribution;
use std::{fmt, mem};

use super::trainer::Trainer;

/// Neural network structure
pub struct Net {
    /// The number of neurons in each layer.
    pub(super) geometry: Box<[usize]>,

    /// Array of coefficients (weights & biases).
    ///
    /// We will store all coefficients of all neurons of all layers here altogether:  
    /// `coeffs = [layer_1][layer_2] ... [layer_N]`  
    /// `layer = [neuron_1][neuron_2] ... [neuron_N]`  
    /// `neuron = [weights]bias`  
    /// So, `coeffs` will look like this:  
    /// `[weights_1]bias_1[weights_2]bias_2 ...`
    pub(super) coeffs: Box<[f64]>,

    /// Buffer for calculating activations of neurons. Used in `Net::process` and `Trainer::train`.
    ///
    /// * When processing, we only need to know activations of previous layer, so we can use two buffers:
    /// `[old activations] & [coefficients] -> calc -> [new activations]`  
    /// Each of these buffers have to be the size of the the biggest layer to fit it when processing.  
    /// We can join this two buffers together:
    /// `buffer = [old activations][new activations]`
    ///
    /// * When training, this buffer will be used for calculating derivatives in the exact same manner:
    /// `buffer = [old derivatives][new derivatives]`
    ///
    /// The resulting buffer will be of the size 2 * _max_layer_size_.
    pub(super) buffer: Box<[f64]>,
}

impl Net {
    /// Returns network for given geometry.  
    /// It will have random coefficients from range [-1,1]
    ///
    /// # Arguments
    /// * `geometry` - a number slice that holds a desired number of neurons in each layer;
    /// * `coefficients` - slice of neurons coefficients (optional).
    ///
    /// # Returns
    /// * `Ok(Net)` if amount of coefficients is right, or no coefficients provided;
    /// * `Err(NewNetError)` otherwise.
    ///
    /// # Examples
    /// * Using random coefficients
    /// ```
    /// # use rpnnet::feedforward::Net;
    /// let mut net = Net::new(&[10, 20, 20, 3], None).unwrap();
    /// ```
    /// * Using given coefficients
    /// ```
    /// # use rpnnet::feedforward::Net;
    /// let coefficients = [0.27, 0.3, 7.5, 0.0, 1.1];
    /// let mut net = Net::new(&[2, 1, 1], Some(Box::new(coefficients))).unwrap();
    /// ```
    pub fn new(geometry: &[usize], coefficients: Option<Box<[f64]>>) -> Result<Net, NewNetError> {
        if geometry.len() < 2 {
            return Err(NewNetError::BadGeometry(geometry.len()));
        }

        //   Calculating array sizes

        // When calculating activations, we would need biggest layer to fit in the Net::buffer.
        // (See Net::buffer's documentation.)
        let mut max_layer_size: usize = 0;

        // Number of all coefficients (weights + biases) of all neurons in all layers.
        // (For Net::coeffs.)
        let mut coeffs_total = 0;

        let mut geometry_iter = geometry.iter();
        let mut old_layer_size = *geometry_iter.next().unwrap(); // Getting 0 (input) layer size

        // Skipped 0 (input) layer, starting from 1
        for &layer_size in geometry_iter {
            coeffs_total += layer_size // Each of [layer_size] neurons has
            * (old_layer_size + 1); // [old_layer_size] weights + 1 bias
            max_layer_size = max_layer_size.max(layer_size);
            old_layer_size = layer_size;
        }

        let coeffs: Box<[f64]> = if let Some(coeffs) = coefficients {
            if coeffs.len() != coeffs_total {
                return Err(NewNetError::BadCoefficients(SizeMismatch {
                    expected: coeffs_total,
                    got: coeffs.len(),
                }));
            }
            coeffs
        } else {
            let mut rng = rand::thread_rng();
            let weights_between = rand::distributions::Uniform::from(-1.0..=1.0);
            let mut coeffs = Vec::with_capacity(coeffs_total);

            let mut geometry_iter = geometry.iter();
            let mut old_layer_size = *geometry_iter.next().unwrap(); // Getting 0 (input) layer size

            // Skipped 0 (input) layer, starting from 1
            for &layer_size in geometry_iter {
                // For each neuron in layer
                for _ in 0..layer_size {
                    // Weights will be random
                    coeffs.extend(weights_between.sample_iter(&mut rng).take(old_layer_size));
                    // But bias will be zero
                    coeffs.push(0.0);
                }
                old_layer_size = layer_size;
            }

            coeffs.into_boxed_slice()
        };

        let mut buffer = Vec::with_capacity(max_layer_size * 2);
        // Filling buffer array with zeros
        buffer.extend((0..max_layer_size * 2).map(|_| 0.0));

        Ok(Net {
            geometry: geometry.to_owned().into_boxed_slice(),
            coeffs: coeffs,
            buffer: buffer.into_boxed_slice(),
        })
    }

    pub fn geometry(&self) -> &[usize] {
        &self.geometry
    }

    /// Exports geometry and coefficients from network.
    ///
    /// # Returns
    /// `(geometry, coefficients)`.
    pub fn export(&self) -> (&[usize], &[f64]) {
        (&self.geometry, &self.coeffs)
    }

    /// Calculates scalar (dot) product of the two given vectors.  
    /// # Arguments
    /// * `a` - first vector;
    /// * `b` - second vector.
    ///
    /// # Returns
    /// * `Ok(f64)` if `a` and `b` have the same lenght;
    /// * `Err(SizeMismatch)` otherwise.
    fn scalar_product(a: &[f64], b: &[f64]) -> Result<f64, SizeMismatch> {
        if a.len() != b.len() {
            return Err(SizeMismatch {
                expected: a.len(),
                got: b.len(),
            });
        };

        Ok(a.iter().zip(b.iter()).map(|(a, b)| a * b).sum())
    }

    /// Sigmoid function.  
    /// Implements the formula:
    /// `1 / (1 + exp(-x))`.
    pub(super) fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Sigmoid derivative function, expressed in terms of sigmoid itself.  
    /// Implements the formula:
    /// `s / (1 - s)`.
    pub(super) fn sigmoid_der_s(s: f64) -> f64 {
        s * (1.0 - s)
    }

    /// Neuron activation function.
    ///
    /// Implements the formula:
    /// `sigmoid((prev_activations . weights) + bias)`.
    ///
    /// # Arguments
    /// * `prev_layer_activations` - slice that holds activations of previous layer;
    /// * `coeffs` - slice that holds weights coefficients & bias (see `Net::coeffs` documentation).
    ///
    /// # Returns
    /// * `Ok(f64)` if `size(coeffs)` = `size(prev_layer_activations)` + `1`(bias);
    /// * `Err(SizeMismatch)` otherwise.
    fn calc_activation(
        prev_layer_activations: &[f64],
        coeffs: &[f64],
    ) -> Result<f64, SizeMismatch> {
        if coeffs.len() != prev_layer_activations.len() + 1 {
            return Err(SizeMismatch {
                expected: prev_layer_activations.len() + 1,
                got: coeffs.len(),
            });
        }

        Ok(Net::sigmoid(
            Net::scalar_product(prev_layer_activations, &coeffs[..coeffs.len() - 1]).unwrap()
                + coeffs.last().unwrap(),
        ))
    }

    /// Used in Net::process and Trainer::train for activations computing iterations.
    ///
    /// # Arguments
    /// * `old_layer_size` - mutable reference to `old_layer_size`;
    /// * `layer_size` - current layer size;
    /// * `remaining_coeffs` - mutable remaining coefficients slice;
    /// * `old_buffer` - slice of old activations buffer;
    /// * `buffer` - slice of current activations mutable buffer.
    pub(super) fn process_act_iteration(
        old_layer_size: &mut usize,
        layer_size: usize,
        remaining_coeffs: &mut &[f64],
        old_buffer: &[f64],
        buffer: &mut [f64],
    ) {
        let mut rem_coeffs = *remaining_coeffs;

        // For each neuron in layer
        for i in 0..layer_size {
            // Getting slice of current coeffs (weights + bias)
            let (current_coeffs, coeffs_tail) = rem_coeffs.split_at(*old_layer_size + 1);
            // And advancing forward
            rem_coeffs = coeffs_tail;

            // Calculating and storing activation for this neuron
            buffer[i] =
                Net::calc_activation(&old_buffer[..*old_layer_size], current_coeffs).unwrap();
        }

        *old_layer_size = layer_size;
        *remaining_coeffs = rem_coeffs;
    }

    /// Calculates output of the network using given input.
    ///
    /// # Arguments
    /// * `inputs` - Slice that holds activations of input neurons;
    /// * `outputs` - Mutable slice that will be filled with activations of output neurons.
    ///
    /// # Returns
    /// * `Ok(())` if amount of imputs size of the outputs is right;
    /// * `Err(ProcessError)` otherwise.
    ///
    /// # Examples
    /// ```
    /// # use rpnnet::feedforward::Net;
    /// let mut net = Net::new(&[10, 20, 20, 3], None).unwrap();
    /// let inputs = [1.0; 10];
    /// let mut outputs = [0.0, 0.0, 0.0];
    /// net.process(&inputs, &mut outputs).unwrap();
    /// ```
    pub fn process(&mut self, inputs: &[f64], outputs: &mut [f64]) -> Result<(), ProcessError> {
        let layers_count = self.geometry.len();

        if inputs.len() != self.geometry[0] {
            return Err(ProcessError::BadInputs(SizeMismatch {
                expected: self.geometry[0],
                got: inputs.len(),
            }));
        }
        if outputs.len() != self.geometry[layers_count - 1] {
            return Err(ProcessError::BadOutputs(SizeMismatch {
                expected: self.geometry[layers_count - 1],
                got: outputs.len(),
            }));
        }

        // Splitting Net::buffer in two, see Net::buffer documentation
        let (mut old_buffer, mut buffer) = self.buffer.split_at_mut(self.buffer.len() / 2);

        let mut remaining_coeffs = self.coeffs.as_ref();

        let (mut old_layer_size, layer_size) = (self.geometry[0], self.geometry[1]);

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
            // First iteration: inputs -> old_buffer
            Net::process_act_iteration(
                &mut old_layer_size,
                layer_size,
                &mut remaining_coeffs,
                inputs,
                old_buffer,
            );

            // Intermediate iterations: old_buffer -> buffer, then swap buffers
            for &layer_size in self.geometry[2..layers_count - 1].iter() {
                Net::process_act_iteration(
                    &mut old_layer_size,
                    layer_size,
                    &mut remaining_coeffs,
                    old_buffer,
                    buffer,
                );
                mem::swap(&mut buffer, &mut old_buffer);
            }

            // Last iteration: old_buffer -> outputs
            Net::process_act_iteration(
                &mut old_layer_size,
                self.geometry[layers_count - 1],
                &mut remaining_coeffs,
                old_buffer,
                outputs,
            );
        }
        Ok(())
    }

    /// Calculates cost function of an output values given the desired values.
    /// Implements the formula:  
    /// `||outputs . desired outputs||`
    ///
    /// # Arguments
    /// * `outputs` - Slice that holds activations of outputs neurons;
    /// * `desired_outputs` - Slice that holds corresponding desired activations.
    ///
    /// # Returns
    /// * `Ok(f64)` if `outputs` and `desired_outputs` have the same size;
    /// * `SizeMismatch` otherwise.
    ///
    /// # Examples
    /// ```
    /// # use rpnnet::feedforward::Net;
    /// let outputs = [10.0; 1000];
    /// let desired_outputs = [10.25; 1000];
    /// let cost = Net::cals_cost(&outputs, &desired_outputs).unwrap();
    /// assert_eq!(cost, 62.5);
    /// ```
    pub fn cals_cost(outputs: &[f64], desired_outputs: &[f64]) -> Result<f64, SizeMismatch> {
        if outputs.len() != desired_outputs.len() {
            return Err(SizeMismatch {
                expected: outputs.len(),
                got: desired_outputs.len(),
            });
        };

        Ok(outputs
            .iter()
            .zip(desired_outputs.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum())
    }

    /// Consumes `Net` and builds `Trainer` object containing it.  
    /// See `Trainer`'s documentation for details.
    pub fn build_trainer(self) -> Trainer {
        Trainer::build(self)
    }
}

/// Error structure for `Net::new`
#[derive(Debug, Clone)]
pub enum NewNetError {
    BadGeometry(usize),
    BadCoefficients(SizeMismatch),
}

impl fmt::Display for NewNetError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            NewNetError::BadGeometry(lenght) => write!(
                f,
                "Net must have at least two layers (input and output), \
                but got geometry with len {}!",
                lenght
            ),
            NewNetError::BadCoefficients(SizeMismatch { expected, got }) => write!(
                f,
                "Expected {} coefficients because of provided geometry, but got {}!",
                expected, got
            ),
        }
    }
}

/// Error structure for `Net::process`
#[derive(Debug, Clone)]
pub enum ProcessError {
    BadInputs(SizeMismatch),
    BadOutputs(SizeMismatch),
}

impl fmt::Display for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            ProcessError::BadInputs(SizeMismatch { expected, got }) => {
                write!(f, "Expected {} input(s), but got {}!", expected, got)
            }
            ProcessError::BadOutputs(SizeMismatch { expected, got }) => {
                write!(f, "Expected {} output(s), but got {}!", expected, got)
            }
        }
    }
}

/// Error structure for collections size mismatch
#[derive(Debug, Clone)]
pub struct SizeMismatch {
    pub expected: usize,
    pub got: usize,
}

impl fmt::Display for SizeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Expected {} values, but got {}!",
            self.expected, self.got
        )
    }
}
