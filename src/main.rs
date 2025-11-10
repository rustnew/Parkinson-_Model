mod neural_network;
mod data;

use neural_network::{NeuralNetwork, Activation, TrainingMetrics};
use data::data_loader::ParkinsonDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 LANCEMENT AVEC CORRECTIONS CRITIQUES");
    
    // 1. CHARGEMENT ET ANALYSE DES DONNÉES
    println!("\n📥 Chargement et analyse des datasets...");
    let mut dataset = ParkinsonDataset::load_all_data()?;
    dataset.shuffle();
    
    // ANALYSE DU DÉSÉQUILIBRE DES CLASSES
    let parkinson_count = dataset.classification_targets.iter()
        .filter(|t| t[0] > 0.5)
        .count();
    let sain_count = dataset.classification_targets.len() - parkinson_count;
    
    println!("📊 Analyse déséquilibre classes:");
    println!("   - Parkinson: {} samples", parkinson_count);
    println!("   - Sain: {} samples", sain_count);
    println!("   - Ratio: {:.1}% vs {:.1}%", 
        (parkinson_count as f64 / dataset.classification_targets.len() as f64) * 100.0,
        (sain_count as f64 / dataset.classification_targets.len() as f64) * 100.0
    );
    
    let stats = dataset.get_stats();
    println!("📈 Dimensions datasets:");
    println!("   - Classification: {} samples ({} features)", stats.classification_samples, stats.classification_features);
    println!("   - Régression: {} samples ({} features)", stats.regression_samples, stats.regression_features);
    
    // 2. ARCHITECTURES CORRIGÉES
    println!("\n🧠 CRÉATION RÉSEAUX CORRIGÉS:");
    
    // CLASSIFICATION CORRIGÉE - Plus de capacité + régularisation
    let mut classification_network = NeuralNetwork::new(0.015); // LR réduit
    classification_network
        .add_layer(22, 64, Activation::Relu)
        .add_layer(64, 32, Activation::Relu)
        .add_layer(32, 16, Activation::Relu)
        .add_layer(16, 1, Activation::Sigmoid);
    
    // RÉGRESSION CORRIGÉE - Plus de capacité
    let mut regression_network = NeuralNetwork::new(0.008); // LR réduit
    regression_network
        .add_layer(16, 128, Activation::Relu)
        .add_layer(128, 64, Activation::Relu)
        .add_layer(64, 32, Activation::Relu)
        .add_layer(32, 1, Activation::Linear);
    
    println!("✅ Classification: 22→64→32→16→1 (4 couches)");
    println!("✅ Régression: 16→128→64→32→1 (4 couches)");
    
    // 3. ENTRAÎNEMENT CORRIGÉ
    println!("\n🎯 ENTRAÎNEMENT CLASSIFICATION CORRIGÉ...");
    let class_metrics = classification_network.train_balanced(
        &dataset.classification_inputs,
        &dataset.classification_targets,
        200,   // Plus d'epochs
        16     // Batch size réduit
    );
    
    println!("\n🎯 ENTRAÎNEMENT RÉGRESSION CORRIGÉ...");
    let reg_metrics = regression_network.train_balanced(
        &dataset.regression_inputs,
        &dataset.regression_targets,
        150,   // Plus d'epochs
        64     // Batch size réduit
    );
    
    // 4. ÉVALUATION CORRECTE
    println!("\n📈 ÉVALUATION CORRIGÉE:");
    
    let (class_loss, class_accuracy, class_precision, class_recall, class_f1) = 
        evaluate_classification_corrected(&classification_network, &dataset);
    
    println!("   - Classification:");
    println!("        Accuracy:  {:.1}%", class_accuracy * 100.0);
    println!("        Precision: {:.1}%", class_precision * 100.0);
    println!("        Recall:    {:.1}%", class_recall * 100.0);
    println!("        F1-Score:  {:.1}%", class_f1 * 100.0);
    println!("        Loss:      {:.6}", class_loss);
    
    let reg_loss = regression_network.evaluate_complete(&dataset.regression_inputs, &dataset.regression_targets);
    println!("   - Régression: Loss={:.6}", reg_loss);
    
    // 5. TESTS COMPLETS
    println!("\n🎯 TESTS COMPLETS CORRIGÉS:");
    test_classification_complete_corrected(&classification_network, &dataset);
    test_regression_complete(&regression_network, &dataset);
    
    // 6. RAPPORT CORRIGÉ
    println!("\n📋 RAPPORT PERFORMANCE CORRIGÉ:");
    generate_corrected_report(&class_metrics, &reg_metrics, class_accuracy, class_precision, class_recall, class_f1);
    
    Ok(())
}

fn evaluate_classification_corrected(
    network: &NeuralNetwork, 
    dataset: &ParkinsonDataset
) -> (f64, f64, f64, f64, f64) {
    let mut true_positives = 0;
    let mut true_negatives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    let mut total_loss = 0.0;
    
    for i in 0..dataset.classification_inputs.len() {
        let prediction = network.forward(&dataset.classification_inputs[i]);
        let target = &dataset.classification_targets[i][0];
        
        // Calcul loss binaire cross-entropy approximative
        let loss = - (target * prediction[0].ln() + (1.0 - target) * (1.0 - prediction[0]).ln());
        total_loss += loss;
        
        let predicted_class = prediction[0] > 0.5;
        let actual_class = *target > 0.5;
        
        match (predicted_class, actual_class) {
            (true, true) => true_positives += 1,
            (false, false) => true_negatives += 1,
            (true, false) => false_positives += 1,
            (false, true) => false_negatives += 1,
        }
    }
    
    let total = dataset.classification_inputs.len();
    let accuracy = (true_positives + true_negatives) as f64 / total as f64;
    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else { 0.0 };
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else { 0.0 };
    let f1_score = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else { 0.0 };
    
    let avg_loss = total_loss / total as f64;
    
    (avg_loss, accuracy, precision, recall, f1_score)
}

fn test_classification_complete_corrected(network: &NeuralNetwork, dataset: &ParkinsonDataset) {
    println!("🧪 TEST CLASSIFICATION COMPLET CORRIGÉ:");
    
    let mut true_positives = 0;
    let mut true_negatives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    
    for i in 0..dataset.classification_inputs.len() {
        let prediction = network.forward(&dataset.classification_inputs[i]);
        let target = &dataset.classification_targets[i][0];
        
        let predicted_class = prediction[0] > 0.5;
        let actual_class = *target > 0.5;
        
        match (predicted_class, actual_class) {
            (true, true) => true_positives += 1,
            (false, false) => true_negatives += 1,
            (true, false) => false_positives += 1,
            (false, true) => false_negatives += 1,
        }
    }
    
    let total = dataset.classification_inputs.len();
    let accuracy = (true_positives + true_negatives) as f64 / total as f64;
    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else { 0.0 };
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else { 0.0 };
    let f1_score = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else { 0.0 };
    
    println!("   📊 Matrice de confusion COMPLÈTE:");
    println!("        Réel/Predi  Parkinson  Sain");
    println!("        Parkinson      {:3}       {:3}", true_positives, false_negatives);
    println!("        Sain           {:3}       {:3}", false_positives, true_negatives);
    println!("   📈 Métriques DÉTAILLÉES:");
    println!("        Accuracy:  {:.1}%", accuracy * 100.0);
    println!("        Precision: {:.1}%", precision * 100.0);
    println!("        Recall:    {:.1}%", recall * 100.0);
    println!("        F1-Score:  {:.1}%", f1_score * 100.0);
    
    // Analyse du déséquilibre
    let total_parkinson = true_positives + false_negatives;
    let total_sain = true_negatives + false_positives;
    println!("   📊 Distribution réelle:");
    println!("        Parkinson: {} samples ({:.1}%)", total_parkinson, (total_parkinson as f64 / total as f64) * 100.0);
    println!("        Sain: {} samples ({:.1}%)", total_sain, (total_sain as f64 / total as f64) * 100.0);
}

fn test_regression_complete(network: &NeuralNetwork, dataset: &ParkinsonDataset) {
    println!("🧪 TEST RÉGRESSION COMPLET:");
    
    let mut total_error = 0.0;
    let test_samples = 1000.min(dataset.regression_inputs.len());
    
    for i in 0..test_samples {
        let prediction = network.forward(&dataset.regression_inputs[i]);
        let target = &dataset.regression_targets[i][0];
        
        let pred_score = prediction[0] * 100.0;
        let real_score = target * 100.0;
        let error = (pred_score - real_score).abs();
        total_error += error;
    }
    
    let avg_error = total_error / test_samples as f64;
    println!("   📊 Erreur moyenne: {:.1} points UPDRS (sur {} samples)", avg_error, test_samples);
}

fn generate_corrected_report(
    class_metrics: &TrainingMetrics, 
    reg_metrics: &TrainingMetrics, 
    accuracy: f64, 
    precision: f64, 
    recall: f64, 
    f1_score: f64
) {
    println!("🎯 PERFORMANCE CORRIGÉE:");
    println!("   🧠 Classification:");
    println!("        - Meilleure loss: {:.6}", class_metrics.best_loss);
    println!("        - Accuracy:  {:.1}%", accuracy * 100.0);
    println!("        - Precision: {:.1}%", precision * 100.0);
    println!("        - Recall:    {:.1}%", recall * 100.0);
    println!("        - F1-Score:  {:.1}%", f1_score * 100.0);
    
    if !class_metrics.losses.is_empty() {
        let improvement = (class_metrics.losses[0] - class_metrics.best_loss) / class_metrics.losses[0] * 100.0;
        println!("        - Amélioration loss: {:.1}%", improvement);
    }
    
    println!("   📈 Régression:");
    println!("        - Meilleure loss: {:.6}", reg_metrics.best_loss);
    
    if !reg_metrics.losses.is_empty() {
        let improvement = (reg_metrics.losses[0] - reg_metrics.best_loss) / reg_metrics.losses[0] * 100.0;
        println!("        - Amélioration: {:.1}%", improvement);
    }
    
    // Score corrigé basé sur F1-score pour classification
    let classification_score = f1_score * 50.0;
    let regression_score = (1.0 - reg_metrics.best_loss) * 50.0;
    let overall_score = classification_score + regression_score;
    
    println!("   🏆 SCORE GLOBAL CORRIGÉ: {:.1}/100", overall_score.min(100.0));
    
    if overall_score > 80.0 {
        println!("   ✅ EXCELLENT: Modèles bien équilibrés!");
    } else if overall_score > 65.0 {
        println!("   ⚡ TRÈS BON: Bonnes performances globales");
    } else if overall_score > 50.0 {
        println!("   ⚠️  CORRECT: Performances acceptables");
    } else {
        println!("   🔄 PROBLÈME: Modèles nécessitent optimisation");
    }
}