use ndarray::Array1;
use rand::seq::SliceRandom;

/// Dataset optimisé pour Parkinson
#[derive(Debug, Clone)]
pub struct ParkinsonDataset {
    pub classification_inputs: Vec<Array1<f64>>,
    pub classification_targets: Vec<Array1<f64>>,
    pub regression_inputs: Vec<Array1<f64>>,
    pub regression_targets: Vec<Array1<f64>>,
}

/// Statistiques optimisées
#[derive(Debug)]
pub struct DataStats {
    pub classification_samples: usize,
    pub regression_samples: usize,
    pub classification_features: usize,
    pub regression_features: usize,
}

impl ParkinsonDataset {
    /// Crée un nouveau dataset vide
    pub fn new() -> Self {
        Self {
            classification_inputs: Vec::new(),
            classification_targets: Vec::new(),
            regression_inputs: Vec::new(),
            regression_targets: Vec::new(),
        }
    }

    /// Charge tous les données rapidement
    pub fn load_all_data() -> Result<Self, Box<dyn std::error::Error>> {
        let mut dataset = Self::new();
        
        println!("📊 Chargement des données Parkinson...");
        
        dataset.load_classification_data()?;
        dataset.load_regression_data()?;
        dataset.normalize_features();
        
        let stats = dataset.get_stats();
        println!("✅ Données chargées: {} class, {} reg", 
            stats.classification_samples, stats.regression_samples);
        
        Ok(dataset)
    }

    /// Charge les données de classification
    fn load_classification_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "parkinsons/parkinsons.data";
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(file_path)?;

        for result in rdr.records() {
            let record = result?;
            
            if record.len() < 24 {
                continue;
            }
            
            let features: Vec<f64> = record.iter()
                .skip(1)
                .take(22)
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            
            if features.len() != 22 {
                continue;
            }
            
            let status: f64 = record.get(23).unwrap_or("0").parse().unwrap_or(0.0);
            
            self.classification_inputs.push(Array1::from_vec(features));
            self.classification_targets.push(Array1::from_vec(vec![status]));
        }
        
        Ok(())
    }

    /// Charge les données de régression
    pub fn load_regression_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "parkinsons/parkinsons_updrs.data";
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(file_path)?;

        for result in rdr.records() {
            let record = result?;
            
            if record.len() < 22 {
                continue;
            }
            
            let features: Vec<f64> = record.iter()
                .skip(6)
                .take(16)
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            
            if features.len() != 16 {
                continue;
            }
            
            let motor_updrs: f64 = record.get(4).unwrap_or("0").parse().unwrap_or(0.0);
            let normalized_updrs = motor_updrs / 100.0;
            
            self.regression_inputs.push(Array1::from_vec(features));
            self.regression_targets.push(Array1::from_vec(vec![normalized_updrs]));
        }
        
        Ok(())
    }

    /// Normalisation rapide
    pub fn normalize_features(&mut self) {
        if !self.classification_inputs.is_empty() {
            Self::normalize_dataset_fast(&mut self.classification_inputs);
        }
        
        if !self.regression_inputs.is_empty() {
            Self::normalize_dataset_fast(&mut self.regression_inputs);
        }
    }

    /// Normalisation optimisée
    fn normalize_dataset_fast(inputs: &mut Vec<Array1<f64>>) {
        if inputs.is_empty() { return; }
        
        let feature_count = inputs[0].len();
        let mut mins = vec![f64::INFINITY; feature_count];
        let mut maxs = vec![f64::NEG_INFINITY; feature_count];
        
        for input in inputs.iter() {
            for (i, &value) in input.iter().enumerate() {
                if value < mins[i] { mins[i] = value; }
                if value > maxs[i] { maxs[i] = value; }
            }
        }
        
        for input in inputs.iter_mut() {
            for i in 0..feature_count {
                let range = maxs[i] - mins[i];
                if range > 0.0 {
                    input[i] = (input[i] - mins[i]) / range;
                }
            }
        }
    }

    /// Statistiques rapides
    pub fn get_stats(&self) -> DataStats {
        DataStats {
            classification_samples: self.classification_inputs.len(),
            regression_samples: self.regression_inputs.len(),
            classification_features: if !self.classification_inputs.is_empty() {
                self.classification_inputs[0].len()
            } else { 0 },
            regression_features: if !self.regression_inputs.is_empty() {
                self.regression_inputs[0].len()
            } else { 0 },
        }
    }

    /// Mélange rapide
    pub fn shuffle(&mut self) {
        let mut rng = rand::rng();
        
        Self::shuffle_dataset(&mut self.classification_inputs, &mut self.classification_targets, &mut rng);
        Self::shuffle_dataset(&mut self.regression_inputs, &mut self.regression_targets, &mut rng);
    }

    fn shuffle_dataset(inputs: &mut Vec<Array1<f64>>, targets: &mut Vec<Array1<f64>>, rng: &mut rand::rngs::ThreadRng) {
        let mut indices: Vec<usize> = (0..inputs.len()).collect();
        indices.shuffle(rng);
        
        let new_inputs: Vec<Array1<f64>> = indices.iter().map(|&i| inputs[i].clone()).collect();
        let new_targets: Vec<Array1<f64>> = indices.iter().map(|&i| targets[i].clone()).collect();
        
        *inputs = new_inputs;
        *targets = new_targets;
    }
    
    pub fn analyze_class_distribution(&self) {
        let mut parkinson_count = 0;
        let mut sain_count = 0;
        
        for target in &self.classification_targets {
            if target[0] > 0.5 {
                parkinson_count += 1;
            } else {
                sain_count += 1;
            }
        }
        
        println!("🔍 ANALYSE DISTRIBUTION CLASSES:");
        println!("   - Parkinson: {} samples", parkinson_count);
        println!("   - Sain: {} samples", sain_count);
        println!("   - Ratio: {:.1}% vs {:.1}%", 
            (parkinson_count as f64 / (parkinson_count + sain_count) as f64) * 100.0,
            (sain_count as f64 / (parkinson_count + sain_count) as f64) * 100.0
        );
        
        if parkinson_count < 10 {
            println!("🚨 ALERTE: Dataset très déséquilibré! Considérez:");
            println!("   - Ajouter plus de données Parkinson");
            println!("   - Utiliser l'augmentation de données");
            println!("   - Techniques de ré-échantillonnage");
        }
    }

}