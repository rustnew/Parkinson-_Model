use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;

pub mod activation;
pub mod layer;
pub mod optimizer;

pub use activation::Activation;
pub use layer::Layer;
pub use optimizer::SGD;

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

/// Entraînement alterné entre classification et régression
pub struct AlternatingTraining {
    pub classification_losses: Vec<f64>,
    pub regression_losses: Vec<f64>,
    pub learning_rates: Vec<f64>,
    pub best_combined_loss: f64,
}

impl AlternatingTraining {
    /// Crée un nouvel entraînement alterné
    pub fn new() -> Self {
        Self {
            classification_losses: Vec::new(),
            regression_losses: Vec::new(),
            learning_rates: Vec::new(),
            best_combined_loss: f64::INFINITY,
        }
    }

    /// Stratégie d'entraînement alterné adaptatif
    pub fn train_alternating(
        network: &mut NeuralNetwork,
        classification_inputs: &[Array1<f64>],
        classification_targets: &[Array1<f64>],
        regression_inputs: &[Array1<f64>],
        regression_targets: &[Array1<f64>],
        epochs: usize,
        batch_size: usize,
    ) -> AlternatingTraining {
        let mut training = AlternatingTraining::new();
        let mut optimizer = SGD::new(network.learning_rate);
        
        println!("🔄 Début entraînement par alternance...");
        println!("   Epochs: {}, Batch size: {}", epochs, batch_size);
        
        for epoch in 0..epochs {
            let mut epoch_classification_loss = 0.0;
            let mut epoch_regression_loss = 0.0;
            
            // STRATÉGIE D'ALTERNANCE ADAPTATIVE
            let strategy = AlternatingTraining::get_training_strategy(epoch, epochs);
            
            // PHASE CLASSIFICATION
            if strategy.classification_ratio > 0.0 {
                let classification_batches = ((classification_inputs.len() as f64 * strategy.classification_ratio) as usize / batch_size).max(1);
                epoch_classification_loss = network.process_phase(
                    classification_inputs, 
                    classification_targets, 
                    &mut optimizer, 
                    classification_batches,
                    batch_size
                );
            }
            
            // PHASE RÉGRESSION
            if strategy.regression_ratio > 0.0 {
                let regression_batches = ((regression_inputs.len() as f64 * strategy.regression_ratio) as usize / batch_size).max(1);
                epoch_regression_loss = network.process_phase(
                    regression_inputs, 
                    regression_targets, 
                    &mut optimizer, 
                    regression_batches,
                    batch_size
                );
            }
            
            // Learning rate adaptatif
            optimizer.learning_rate = AlternatingTraining::adaptive_learning_rate(
                epoch, 
                epoch_classification_loss + epoch_regression_loss,
                optimizer.learning_rate
            );
            
            training.classification_losses.push(epoch_classification_loss);
            training.regression_losses.push(epoch_regression_loss);
            training.learning_rates.push(optimizer.learning_rate);
            
            let combined_loss = epoch_classification_loss + epoch_regression_loss;
            if combined_loss < training.best_combined_loss {
                training.best_combined_loss = combined_loss;
            }
            
            // Affichage progression
            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!("Epoch {:4} | Class: {:.6} | Reg: {:.6} | LR: {:.6} | Strat: {:.1}:{:.1}", 
                    epoch, epoch_classification_loss, epoch_regression_loss, 
                    optimizer.learning_rate, strategy.classification_ratio, strategy.regression_ratio);
            }
        }
        
        println!("✅ Entraînement alternance terminé! Meilleure loss combinée: {:.6}", training.best_combined_loss);
        training
    }

    /// Détermine la stratégie d'alternance selon la progression
    fn get_training_strategy(epoch: usize, total_epochs: usize) -> TrainingStrategy {
        let progress = epoch as f64 / total_epochs as f64;
        
        if progress < 0.3 {
            // Phase 1: Plus de régression pour apprendre les patterns
            TrainingStrategy { classification_ratio: 0.3, regression_ratio: 0.7 }
        } else if progress < 0.7 {
            // Phase 2: Équilibre
            TrainingStrategy { classification_ratio: 0.5, regression_ratio: 0.5 }
        } else {
            // Phase 3: Affinage classification
            TrainingStrategy { classification_ratio: 0.7, regression_ratio: 0.3 }
        }
    }

    /// Ajuste le learning rate de façon adaptative
    fn adaptive_learning_rate(epoch: usize, _current_loss: f64, current_lr: f64) -> f64 {
        // Réduction progressive
        if epoch > 0 && epoch % 50 == 0 {
            current_lr * 0.9
        } else {
            current_lr
        }.max(1e-6)
    }
}

/// Stratégie de répartition entre classification et régression
struct TrainingStrategy {
    classification_ratio: f64,
    regression_ratio: f64,
}

/// Réseau neuronal avec entraînement et évaluation
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

    /// Entraîne le réseau sur les données fournies
    pub fn train(
        &mut self, 
        inputs: &[Array1<f64>], 
        targets: &[Array1<f64>], 
        epochs: usize,
        batch_size: usize,
    ) -> TrainingMetrics {
        let mut optimizer = SGD::new(self.learning_rate);
        let mut current_lr = self.learning_rate;
        
        println!("🎯 Début de l'entraînement avec {} échantillons...", inputs.len());
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut batches_processed = 0;
            
            // Mélange des données
            let mut indices: Vec<usize> = (0..inputs.len()).collect();
            Self::shuffle_indices(&mut indices);
            
            for batch_start in (0..inputs.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(inputs.len());
                let batch_indices = &indices[batch_start..batch_end];
                
                let batch_inputs: Vec<Array1<f64>> = batch_indices.iter()
                    .map(|&i| inputs[i].clone())
                    .collect();
                let batch_targets: Vec<Array1<f64>> = batch_indices.iter()
                    .map(|&i| targets[i].clone())
                    .collect();
                
                let batch_loss = self.process_batch(&batch_inputs, &batch_targets, &mut optimizer);
                epoch_loss += batch_loss;
                batches_processed += 1;
            }
            
            if batches_processed > 0 {
                let avg_loss = epoch_loss / batches_processed as f64;
                
                // Learning rate adaptatif simple
                current_lr = self.adaptive_learning_rate(epoch, current_lr);
                optimizer.learning_rate = current_lr;
                
                let improved = self.metrics.update(avg_loss, 0.0, current_lr);
                self.print_training_progress(epoch, avg_loss, current_lr, improved);
                
                // Early stopping
                if self.metrics.patience_counter > 100 {
                    println!("🛑 Early stopping à l'epoch {} (pas d'amélioration depuis 100 epochs)", epoch);
                    break;
                }
            }
        }
        
        println!("✅ Entraînement terminé! Meilleure loss: {:.6}", self.metrics.best_loss);
        self.metrics.clone()
    }

    /// Traite une phase d'entraînement (classification ou régression)
    pub fn process_phase(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[Array1<f64>],
        optimizer: &mut SGD,
        num_batches: usize,
        batch_size: usize,
    ) -> f64 {
        let mut total_loss = 0.0;
        let mut batches_processed = 0;
        
        for _ in 0..num_batches {
            if inputs.is_empty() { break; }
            
            // Sélection aléatoire dans le dataset
            let batch_inputs: Vec<Array1<f64>> = (0..batch_size)
                .filter_map(|_| {
                    let idx = (rand::random::<f64>() * inputs.len() as f64) as usize;
                    if idx < inputs.len() {
                        Some(inputs[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();
                
            let batch_targets: Vec<Array1<f64>> = (0..batch_size)
                .filter_map(|_| {
                    let idx = (rand::random::<f64>() * targets.len() as f64) as usize;
                    if idx < targets.len() {
                        Some(targets[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();
            
            if batch_inputs.is_empty() { continue; }
            
            let batch_loss = self.process_batch(&batch_inputs, &batch_targets, optimizer);
            total_loss += batch_loss;
            batches_processed += 1;
        }
        
        if batches_processed > 0 { total_loss / batches_processed as f64 } else { 0.0 }
    }

    /// Traite un batch de données (propagation avant + rétropropagation)
    fn process_batch(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[Array1<f64>],
        optimizer: &mut SGD,
    ) -> f64 {
        let mut total_loss = 0.0;
        let batch_size = inputs.len();
        
        let mut weight_gradients: Vec<Array2<f64>> = self.layers.iter()
            .map(|layer| Array2::zeros((layer.output_size, layer.input_size)))
            .collect();
            
        let mut bias_gradients: Vec<Array1<f64>> = self.layers.iter()
            .map(|layer| Array1::zeros(layer.output_size))
            .collect();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let (output, activations) = self.forward_with_cache(input);
            let loss = self.mse_loss(&output, target);
            total_loss += loss;
            
            let gradients = self.backward(&output, target, &activations);
            
            for (i, (wg, bg)) in gradients.iter().enumerate() {
                weight_gradients[i] = &weight_gradients[i] + wg;
                bias_gradients[i] = &bias_gradients[i] + bg;
            }
        }

        // Gradient Clipping doux
        self.gradient_clipping(&mut weight_gradients, &mut bias_gradients, 3.0);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let avg_weight_grad = &weight_gradients[i] / batch_size as f64;
            let avg_bias_grad = &bias_gradients[i] / batch_size as f64;
            
            layer.weights = optimizer.update_weights(&layer.weights, &avg_weight_grad);
            layer.biases = optimizer.update_biases(&layer.biases, &avg_bias_grad);
        }

        total_loss / batch_size as f64
    }

    /// Applique le gradient clipping pour stabiliser l'entraînement
    pub fn gradient_clipping(&self, weight_grads: &mut [Array2<f64>], bias_grads: &mut [Array1<f64>], max_norm: f64) {
        let weight_norm: f64 = weight_grads.iter()
            .map(|grad| grad.mapv(|x| x.powi(2)).sum())
            .sum::<f64>();
        
        let bias_norm: f64 = bias_grads.iter()
            .map(|grad| grad.mapv(|x| x.powi(2)).sum())
            .sum::<f64>();
            
        let total_norm = (weight_norm + bias_norm).sqrt();

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

    /// Ajuste le learning rate de façon adaptative
    fn adaptive_learning_rate(&self, epoch: usize, current_lr: f64) -> f64 {
        // Réduction progressive toutes les 50 epochs
        if epoch > 0 && epoch % 50 == 0 {
            current_lr * 0.95
        } else {
            current_lr
        }.max(1e-6) // LR minimum
    }

    /// Mélange les indices pour l'entraînement
    pub fn shuffle_indices(indices: &mut [usize]) {
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
    }

    /// Affiche la progression de l'entraînement
    pub fn print_training_progress(&self, epoch: usize, loss: f64, lr: f64, improved: bool) {
        let marker = if improved { "✨" } else { "  " };
        if epoch % 20 == 0 || improved || epoch < 10 {
            println!("Epoch {:4} {} Loss: {:.6} | LR: {:.6} {}", 
                epoch, marker, loss, lr,
                if improved { "📈" } else { "" });
        }
    }

    /// Évalue les performances du réseau sur un dataset de test
    pub fn evaluate(&self, inputs: &[Array1<f64>], targets: &[Array1<f64>]) -> f64 {
        let mut total_loss = 0.0;
        
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = self.forward(input);
            let loss = self.mse_loss(&output, target);
            total_loss += loss;
        }
        
        total_loss / inputs.len() as f64
    }

    /// Analyse détaillée des performances
    pub fn analyze_performance(&self) {
        println!("\n📊 ANALYSE DES PERFORMANCES:");
        println!("Meilleure loss: {:.6}", self.metrics.best_loss);
        if let Some(last_loss) = self.metrics.losses.last() {
            println!("Dernière loss: {:.6}", last_loss);
            if let Some(first_loss) = self.metrics.losses.first() {
                let improvement = (first_loss - last_loss) / first_loss * 100.0;
                println!("Amélioration: {:.2}%", improvement);
            }
        }
        
        if let Some((min_idx, min_loss)) = self.metrics.losses.iter().enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            println!("Meilleure epoch: {} (loss: {:.6})", min_idx, min_loss);
        }
    }

    /// Propagation avant avec cache des activations pour la rétropropagation
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

    /// Rétropropagation du gradient
    pub fn backward(
        &self,
        output: &Array1<f64>,
        target: &Array1<f64>,
        activations: &[(Array1<f64>, Array1<f64>)],
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let mut gradients = Vec::new();
        let mut delta = output - target;

        for (i, layer) in self.layers.iter().enumerate().rev() {
            let (input, z) = &activations[i];
            
            let activation_derivative = layer.activation.derivative(z);
            delta = &delta * &activation_derivative;
            
            let weight_gradient = {
                let delta_2d = delta.view().insert_axis(ndarray::Axis(1));
                let input_2d = input.view().insert_axis(ndarray::Axis(0));
                delta_2d.dot(&input_2d)
            };
                
            let bias_gradient = delta.clone();
            
            gradients.push((weight_gradient, bias_gradient));
            
            if i > 0 {
                delta = layer.weights.t().dot(&delta);
            }
        }
        
        gradients.reverse();
        gradients
    }
}