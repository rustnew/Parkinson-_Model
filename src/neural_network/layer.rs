use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use super::Activation;

/// Couche de neurones optimisée
#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation: Activation,
    pub input_size: usize,
    pub output_size: usize,
}

impl Layer {
    /// Crée une nouvelle couche avec initialisation optimisée
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::rng();
        let std_dev = (2.0 / input_size as f64).sqrt();
        
        // Initialisation He optimisée
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.random_range(-std_dev..std_dev)
        });
        
        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation,
            input_size,
            output_size,
        }
    }

    /// Propagation avant optimisée
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = &self.weights.dot(input) + &self.biases;
        self.activation.activate(&z)
    }
}

// Implémentations de sérialisation pour la persistance (optionnel)
#[derive(Serialize, Deserialize)]
struct LayerData {
    weights: Vec<f64>,
    biases: Vec<f64>,
    weights_shape: (usize, usize),
    activation: Activation,
    input_size: usize,
    output_size: usize,
}

impl Serialize for Layer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let data = LayerData {
            weights: self.weights.iter().cloned().collect(),
            biases: self.biases.iter().cloned().collect(),
            weights_shape: self.weights.dim(),
            activation: self.activation.clone(),
            input_size: self.input_size,
            output_size: self.output_size,
        };
        data.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Layer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = LayerData::deserialize(deserializer)?;
        
        let weights = Array2::from_shape_vec(data.weights_shape, data.weights)
            .map_err(serde::de::Error::custom)?;
        let biases = Array1::from_vec(data.biases);
        
        Ok(Layer {
            weights,
            biases,
            activation: data.activation,
            input_size: data.input_size,
            output_size: data.output_size,
        })
    }
}