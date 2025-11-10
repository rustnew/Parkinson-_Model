use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Fonctions d'activation pour les réseaux neuronaux
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Linear,
    Softmax,
}

impl Activation {
    /// Applique la fonction d'activation
    pub fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Relu => self.relu(x),
            Self::Sigmoid => self.sigmoid(x),
            Self::Tanh => self.tanh(x),
            Self::Linear => x.clone(),
            Self::Softmax => self.softmax(x),
        }
    }

    /// Calcule la dérivée de la fonction d'activation
    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Relu => self.relu_derivative(x),
            Self::Sigmoid => {
                let sigmoid = self.sigmoid(x);
                &sigmoid * &(1.0 - &sigmoid)
            }
            Self::Tanh => {
                let tanh = self.tanh(x);
                1.0 - tanh.mapv(|x| x.powi(2))
            }
            Self::Linear => Array1::ones(x.len()),
            Self::Softmax => self.softmax_derivative(x),
        }
    }

    /// Fonction ReLU: max(0, x)
    fn relu(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }

    /// Dérivée de ReLU
    fn relu_derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    /// Fonction sigmoid: 1 / (1 + exp(-x))
    fn sigmoid(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Fonction tanh
    fn tanh(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.tanh())
    }

    /// Fonction softmax pour la classification multi-classes
    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Array1<f64> = x.mapv(|v| (v - max).exp());
        let sum: f64 = exp.sum();
        exp / sum
    }

    /// Dérivée de softmax
    fn softmax_derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let softmax = self.softmax(x);
        &softmax * &(1.0 - &softmax)
    }
}