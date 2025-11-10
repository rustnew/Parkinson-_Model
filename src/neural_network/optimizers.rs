use ndarray::{Array1, Array2};

/// Optimiseur SGD (Stochastic Gradient Descent)
pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    /// Crée un nouvel optimiseur SGD
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    /// Met à jour les poids selon la règle de descente de gradient
    pub fn update_weights(&self, weights: &Array2<f64>, gradients: &Array2<f64>) -> Array2<f64> {
        weights - &(gradients * self.learning_rate)
    }

    /// Met à jour les biais selon la règle de descente de gradient
    pub fn update_biases(&self, biases: &Array1<f64>, gradients: &Array1<f64>) -> Array1<f64> {
        biases - &(gradients * self.learning_rate)
    }
}