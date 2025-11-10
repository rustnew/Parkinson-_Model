pub mod activation;
pub mod layer;
pub mod optimizers;

pub use activation::Activation;
pub use layer::Layer;
pub use optimizers::SGD;

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;

/// Métriques de suivi pendant l'entraînement
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub losses: Vec<f64>,
    pub gradients_norm: Vec<f64>,
    pub learning_rates: Vec<f64>,
    pub best_loss: f64,
    pub patience_counter: usize,
}

impl TrainingMetrics {
    /// Crée de nouvelles métriques d'entraînement
    pub fn new() -> Self {
        Self {
            losses: Vec::new(),
            gradients_norm: Vec::new(),
            learning_rates: Vec::new(),
            best_loss: f64::INFINITY,
            patience_counter: 0,
        }
    }

    /// Met à jour les métriques avec les nouvelles valeurs
    pub fn update(&mut self, loss: f64, grad_norm: f64, lr: f64) -> bool {
        self.losses.push(loss);
        self.gradients_norm.push(grad_norm);
        self.learning_rates.push(lr);

        let improved = loss < self.best_loss;
        if improved {
            self.best_loss = loss;
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
        }

        improved
    }
}

/// Réseau neuronal optimisé
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    learning_rate: f64,
    metrics: TrainingMetrics,
}

impl NeuralNetwork {
    /// Crée un nouveau réseau neuronal
    pub fn new(learning_rate: f64) -> Self {
        Self {
            layers: Vec::new(),
            learning_rate,
            metrics: TrainingMetrics::new(),
        }
    }

    /// Ajoute une couche au réseau
    pub fn add_layer(&mut self, input_size: usize, output_size: usize, activation: Activation) -> &mut Self {
        let layer = Layer::new(input_size, output_size, activation);
        self.layers.push(layer);
        self
    }

    /// Propagation avant à travers tout le réseau
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    /// ENTRAÎNEMENT ULTRA RAPIDE
    pub fn train_fast(
        &mut self, 
        inputs: &[Array1<f64>], 
        targets: &[Array1<f64>], 
        epochs: usize,
        batch_size: usize,
    ) -> TrainingMetrics {
        let mut optimizer = SGD::new(self.learning_rate);
        
        println!("⚡ Entraînement rapide - {} samples, batch: {}", inputs.len(), batch_size);
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut batches_processed = 0;
            
            // Mélange optimisé
            let mut indices: Vec<usize> = (0..inputs.len()).collect();
            Self::shuffle_indices_fast(&mut indices);
            
            for batch_start in (0..inputs.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(inputs.len());
                let batch_loss = self.process_batch_ultra_fast(
                    inputs, 
                    targets, 
                    &indices[batch_start..batch_end],
                    &mut optimizer
                );
                epoch_loss += batch_loss;
                batches_processed += 1;
            }
            
            if batches_processed > 0 {
                let avg_loss = epoch_loss / batches_processed as f64;
                
                // Learning rate adaptatif agressif
                optimizer.learning_rate = self.aggressive_learning_rate(epoch, avg_loss, optimizer.learning_rate);
                
                let improved = self.metrics.update(avg_loss, 0.0, optimizer.learning_rate);
                
                // Affichage minimal pour performance
                if epoch % 20 == 0 || epoch == epochs - 1 || improved {
                    let marker = if improved { "🚀" } else { "  " };
                    println!("Epoch {:3} {} Loss: {:.6}", epoch, marker, avg_loss);
                }
                
                // Early stopping agressif
                if self.metrics.patience_counter > 30 {
                    println!("⏹️  Arrêt early à epoch {}", epoch);
                    break;
                }
            }
        }
        
        println!("✅ Entraînement terminé! Best loss: {:.6}", self.metrics.best_loss);
        self.metrics.clone()
    }

    /// ENTRAÎNEMENT AVEC GESTION DU DÉSÉQUILIBRE
    pub fn train_balanced(
        &mut self, 
        inputs: &[Array1<f64>], 
        targets: &[Array1<f64>], 
        epochs: usize,
        batch_size: usize,
    ) -> TrainingMetrics {
        let mut optimizer = SGD::new(self.learning_rate);
        
        println!("🎯 Entraînement équilibré - {} samples", inputs.len());
        println!("   Batch size: {}, Epochs: {}", batch_size, epochs);
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut batches_processed = 0;
            
            let mut indices: Vec<usize> = (0..inputs.len()).collect();
            Self::shuffle_indices_fast(&mut indices);
            
            for batch_start in (0..inputs.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(inputs.len());
                let batch_loss = self.process_batch_balanced(
                    inputs, 
                    targets, 
                    &indices[batch_start..batch_end],
                    &mut optimizer
                );
                epoch_loss += batch_loss;
                batches_processed += 1;
            }
            
            if batches_processed > 0 {
                let avg_loss = epoch_loss / batches_processed as f64;
                
                // Learning rate adaptatif plus conservateur
                optimizer.learning_rate = self.conservative_learning_rate(epoch, avg_loss, optimizer.learning_rate);
                
                let improved = self.metrics.update(avg_loss, 0.0, optimizer.learning_rate);
                
                if epoch % 20 == 0 || epoch == epochs - 1 || improved {
                    let marker = if improved { "📈" } else { "  " };
                    println!("Epoch {:3} {} Loss: {:.6} | LR: {:.6}", 
                        epoch, marker, avg_loss, optimizer.learning_rate);
                }
                
                // Early stopping plus patient
                if self.metrics.patience_counter > 50 {
                    println!("⏹️  Arrêt équilibré à epoch {}", epoch);
                    break;
                }
            }
        }
        
        println!("✅ Entraînement équilibré terminé! Best loss: {:.6}", self.metrics.best_loss);
        self.metrics.clone()
    }

    /// Traitement de batch ultra rapide
    fn process_batch_ultra_fast(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[Array1<f64>],
        batch_indices: &[usize],
        optimizer: &mut SGD,
    ) -> f64 {
        let batch_size = batch_indices.len();
        let mut total_loss = 0.0;
        
        // Pré-allocation des gradients
        let mut weight_gradients: Vec<Array2<f64>> = self.layers.iter()
            .map(|layer| Array2::zeros((layer.output_size, layer.input_size)))
            .collect();
            
        let mut bias_gradients: Vec<Array1<f64>> = self.layers.iter()
            .map(|layer| Array1::zeros(layer.output_size))
            .collect();

        // Traitement vectorisé
        for &idx in batch_indices {
            let input = &inputs[idx];
            let target = &targets[idx];
            
            let (output, activations) = self.forward_with_cache(input);
            total_loss += self.mse_loss(&output, target);
            
            let gradients = self.backward_fast(&output, target, &activations);
            
            for (i, (wg, bg)) in gradients.iter().enumerate() {
                weight_gradients[i] = &weight_gradients[i] + wg;
                bias_gradients[i] = &bias_gradients[i] + bg;
            }
        }

        // Mise à jour des poids
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let avg_weight_grad = &weight_gradients[i] / batch_size as f64;
            let avg_bias_grad = &bias_gradients[i] / batch_size as f64;
            
            layer.weights = optimizer.update_weights(&layer.weights, &avg_weight_grad);
            layer.biases = optimizer.update_biases(&layer.biases, &avg_bias_grad);
        }

        total_loss / batch_size as f64
    }

    /// Traitement de batch équilibré
    fn process_batch_balanced(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[Array1<f64>],
        batch_indices: &[usize],
        optimizer: &mut SGD,
    ) -> f64 {
        let batch_size = batch_indices.len();
        let mut total_loss = 0.0;
        
        let mut weight_gradients: Vec<Array2<f64>> = self.layers.iter()
            .map(|layer| Array2::zeros((layer.output_size, layer.input_size)))
            .collect();
            
        let mut bias_gradients: Vec<Array1<f64>> = self.layers.iter()
            .map(|layer| Array1::zeros(layer.output_size))
            .collect();

        for &idx in batch_indices {
            let input = &inputs[idx];
            let target = &targets[idx];
            
            let (output, activations) = self.forward_with_cache(input);
            
            // Perte avec régularisation implicite pour équilibrage
            let loss = self.balanced_loss(&output, target);
            total_loss += loss;
            
            let gradients = self.backward_balanced(&output, target, &activations);
            
            for (i, (wg, bg)) in gradients.iter().enumerate() {
                weight_gradients[i] = &weight_gradients[i] + wg;
                bias_gradients[i] = &bias_gradients[i] + bg;
            }
        }

        // Gradient clipping pour stabilité
        self.optimal_gradient_clipping(&mut weight_gradients, &mut bias_gradients, 2.0);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let avg_weight_grad = &weight_gradients[i] / batch_size as f64;
            let avg_bias_grad = &bias_gradients[i] / batch_size as f64;
            
            layer.weights = optimizer.update_weights(&layer.weights, &avg_weight_grad);
            layer.biases = optimizer.update_biases(&layer.biases, &avg_bias_grad);
        }

        total_loss / batch_size as f64
    }

    /// Perte équilibrée pour gérer le déséquilibre des classes
    fn balanced_loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        // MSE avec pondération pour équilibrer les classes
        let base_loss = self.mse_loss(output, target);
        
        // Pénalité supplémentaire pour les faux négatifs (cas importants manqués)
        if target[0] > 0.5 && output[0] < 0.3 {
            base_loss * 2.0  // Double pénalité pour faux négatifs
        } else {
            base_loss
        }
    }

    /// Rétropropagation optimisée
    fn backward_fast(
        &self,
        output: &Array1<f64>,
        target: &Array1<f64>,
        activations: &[(Array1<f64>, Array1<f64>)],
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let mut gradients = Vec::new();
        let mut delta = output - target;

        for (i, layer) in self.layers.iter().enumerate().rev() {
            let (ref input, ref z) = activations[i];
            
            let activation_derivative = layer.activation.derivative(z);
            delta = &delta * &activation_derivative;
            
            let weight_gradient = {
                let delta_2d = delta.view().insert_axis(ndarray::Axis(1));
                let input_2d = input.view().insert_axis(ndarray::Axis(0));
                delta_2d.dot(&input_2d)
            };
                
            gradients.push((weight_gradient, delta.clone()));
            
            if i > 0 {
                delta = layer.weights.t().dot(&delta);
            }
        }
        
        gradients.reverse();
        gradients
    }

    /// Rétropropagation équilibrée
    fn backward_balanced(
        &self,
        output: &Array1<f64>,
        target: &Array1<f64>,
        activations: &[(Array1<f64>, Array1<f64>)],
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let mut gradients = Vec::new();
        let mut delta = output - target;

        // Renforcement des gradients pour les cas positifs (Parkinson)
        if target[0] > 0.5 {
            delta = &delta * 1.5; // Augmente l'importance des cas Parkinson
        }

        for (i, layer) in self.layers.iter().enumerate().rev() {
            let (ref input, ref z) = activations[i];
            
            let activation_derivative = layer.activation.derivative(z);
            delta = &delta * &activation_derivative;
            
            let weight_gradient = {
                let delta_2d = delta.view().insert_axis(ndarray::Axis(1));
                let input_2d = input.view().insert_axis(ndarray::Axis(0));
                delta_2d.dot(&input_2d)
            };
                
            gradients.push((weight_gradient, delta.clone()));
            
            if i > 0 {
                delta = layer.weights.t().dot(&delta);
            }
        }
        
        gradients.reverse();
        gradients
    }

    /// Learning rate agressif pour convergence rapide
    fn aggressive_learning_rate(&self, epoch: usize, _current_loss: f64, current_lr: f64) -> f64 {
        if epoch < 10 {
            // Phase initiale: LR élevé
            current_lr
        } else if epoch < 30 {
            // Phase de convergence: LR modéré
            current_lr * 0.95
        } else {
            // Phase finale: LR bas
            current_lr * 0.9
        }.max(1e-4) // Minimum plus élevé pour éviter stagnation
    }

    /// Learning rate conservateur pour stabilité
    fn conservative_learning_rate(&self, _epoch: usize, _current_loss: f64, current_lr: f64) -> f64 {
        if _epoch < 30 {
            current_lr  // Phase initiale stable
        } else if _epoch < 100 {
            current_lr * 0.98  // Réduction lente
        } else {
            current_lr * 0.95  // Réduction modérée
        }.max(1e-6)  // Minimum très bas
    }

    /// Gradient clipping optimal
    fn optimal_gradient_clipping(&self, weight_grads: &mut [Array2<f64>], bias_grads: &mut [Array1<f64>], max_norm: f64) {
        let total_norm: f64 = weight_grads.iter()
            .map(|grad| grad.mapv(|x| x.powi(2)).sum())
            .sum::<f64>()
            + bias_grads.iter()
                .map(|grad| grad.mapv(|x| x.powi(2)).sum())
                .sum::<f64>();
        
        let total_norm = total_norm.sqrt();

        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for grad in weight_grads {
                *grad = grad.mapv(|x| x * scale);
            }
            for grad in bias_grads {
                *grad = grad.mapv(|x| x * scale);
            }
        }
    }

    /// Mélange ultra rapide
    pub fn shuffle_indices_fast(indices: &mut [usize]) {
        let mut rng = rand::rng();
        indices.shuffle(&mut rng);
    }

    /// Évaluation rapide
    pub fn evaluate_fast(&self, inputs: &[Array1<f64>], targets: &[Array1<f64>], max_samples: usize) -> f64 {
        let test_size = inputs.len().min(max_samples);
        let mut total_loss = 0.0;
        
        for i in 0..test_size {
            let output = self.forward(&inputs[i]);
            total_loss += self.mse_loss(&output, &targets[i]);
        }
        
        total_loss / test_size as f64
    }

    /// Évaluation complète
    pub fn evaluate_complete(&self, inputs: &[Array1<f64>], targets: &[Array1<f64>]) -> f64 {
        let mut total_loss = 0.0;
        
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = self.forward(input);
            total_loss += self.mse_loss(&output, target);
        }
        
        total_loss / inputs.len() as f64
    }

    /// Propagation avant avec cache pour la rétropropagation
    pub fn forward_with_cache(&self, input: &Array1<f64>) -> (Array1<f64>, Vec<(Array1<f64>, Array1<f64>)>) {
        let mut activations = Vec::new();
        let mut current_activation = input.clone();
        
        for layer in &self.layers {
            let z = &layer.weights.dot(&current_activation) + &layer.biases;
            activations.push((current_activation.clone(), z.clone()));
            current_activation = layer.activation.activate(&z);
        }
        
        (current_activation, activations)
    }

    /// Calcule la loss MSE (Mean Squared Error)
    pub fn mse_loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        output.iter().zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f64>() / output.len() as f64
    }

    /// Entraînement avec paramètres optimaux
    pub fn train_optimal(
        &mut self, 
        inputs: &[Array1<f64>], 
        targets: &[Array1<f64>], 
        epochs: usize,
        batch_size: usize,
    ) -> TrainingMetrics {
        let mut optimizer = SGD::new(self.learning_rate);
        
        println!("🎯 Entraînement optimal - {} samples", inputs.len());
        println!("   Architecture: {} couches", self.layers.len());
        println!("   Paramètres: {} epochs, batch: {}", epochs, batch_size);
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut batches_processed = 0;
            
            let mut indices: Vec<usize> = (0..inputs.len()).collect();
            Self::shuffle_indices_fast(&mut indices);
            
            for batch_start in (0..inputs.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(inputs.len());
                let batch_loss = self.process_batch_optimal(
                    inputs, 
                    targets, 
                    &indices[batch_start..batch_end],
                    &mut optimizer
                );
                epoch_loss += batch_loss;
                batches_processed += 1;
            }
            
            if batches_processed > 0 {
                let avg_loss = epoch_loss / batches_processed as f64;
                
                // Learning rate adaptatif optimal
                optimizer.learning_rate = self.optimal_learning_rate(epoch, avg_loss, optimizer.learning_rate);
                
                let improved = self.metrics.update(avg_loss, 0.0, optimizer.learning_rate);
                
                if epoch % 25 == 0 || epoch == epochs - 1 || improved {
                    let marker = if improved { "📈" } else { "  " };
                    println!("Epoch {:3} {} Loss: {:.6} | LR: {:.5}", 
                        epoch, marker, avg_loss, optimizer.learning_rate);
                }
                
                // Early stopping optimal
                if self.metrics.patience_counter > 35 {
                    println!("⏹️  Arrêt optimal à epoch {}", epoch);
                    break;
                }
            }
        }
        
        println!("✅ Entraînement optimal terminé! Best loss: {:.6}", self.metrics.best_loss);
        self.metrics.clone()
    }

    /// Traitement de batch optimal
    fn process_batch_optimal(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[Array1<f64>],
        batch_indices: &[usize],
        optimizer: &mut SGD,
    ) -> f64 {
        let batch_size = batch_indices.len();
        let mut total_loss = 0.0;
        
        let mut weight_gradients: Vec<Array2<f64>> = self.layers.iter()
            .map(|layer| Array2::zeros((layer.output_size, layer.input_size)))
            .collect();
            
        let mut bias_gradients: Vec<Array1<f64>> = self.layers.iter()
            .map(|layer| Array1::zeros(layer.output_size))
            .collect();

        for &idx in batch_indices {
            let input = &inputs[idx];
            let target = &targets[idx];
            
            let (output, activations) = self.forward_with_cache(input);
            total_loss += self.mse_loss(&output, target);
            
            let gradients = self.backward_optimal(&output, target, &activations);
            
            for (i, (wg, bg)) in gradients.iter().enumerate() {
                weight_gradients[i] = &weight_gradients[i] + wg;
                bias_gradients[i] = &bias_gradients[i] + bg;
            }
        }

        // Gradient clipping optimal
        self.optimal_gradient_clipping(&mut weight_gradients, &mut bias_gradients, 2.5);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let avg_weight_grad = &weight_gradients[i] / batch_size as f64;
            let avg_bias_grad = &bias_gradients[i] / batch_size as f64;
            
            layer.weights = optimizer.update_weights(&layer.weights, &avg_weight_grad);
            layer.biases = optimizer.update_biases(&layer.biases, &avg_bias_grad);
        }

        total_loss / batch_size as f64
    }

    /// Rétropropagation optimale
    fn backward_optimal(
        &self,
        output: &Array1<f64>,
        target: &Array1<f64>,
        activations: &[(Array1<f64>, Array1<f64>)],
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let mut gradients = Vec::new();
        let mut delta = output - target;

        for (i, layer) in self.layers.iter().enumerate().rev() {
            let (ref input, ref z) = activations[i];
            
            let activation_derivative = layer.activation.derivative(z);
            delta = &delta * &activation_derivative;
            
            let weight_gradient = {
                let delta_2d = delta.view().insert_axis(ndarray::Axis(1));
                let input_2d = input.view().insert_axis(ndarray::Axis(0));
                delta_2d.dot(&input_2d)
            };
                
            gradients.push((weight_gradient, delta.clone()));
            
            if i > 0 {
                delta = layer.weights.t().dot(&delta);
            }
        }
        
        gradients.reverse();
        gradients
    }

    /// Learning rate optimal
    fn optimal_learning_rate(&self, epoch: usize, _current_loss: f64, current_lr: f64) -> f64 {
        match epoch {
            0..=20 => current_lr,                    // Phase initiale stable
            21..=60 => current_lr * 0.97,            // Réduction progressive
            61..=90 => current_lr * 0.95,            // Réduction accélérée
            _ => current_lr * 0.92,                  // Phase finale
        }.max(1e-5) // Minimum optimal
    }
}