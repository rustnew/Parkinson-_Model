<img width="1305" height="643" alt="image" src="https://github.com/user-attachments/assets/7087e073-e794-4c70-9eee-0ecc55269bc7" />
# 🧠 Parkinson Disease Detection & Progression Monitoring - Neural Network in Rust

## 📖 Description du Projet

Ce projet implémente un **réseau de neurones profond multi-tâches** en Rust pour la détection et le suivi de la maladie de Parkinson. Le modèle combine deux approches complémentaires : la **classification binaire** (diagnostic) et la **régression** (évaluation de sévérité) à partir de mesures vocales biomédicales.

## 🎯 Objectifs du Projet

### **Principal**
Développer un système intelligent capable de :
- 🔍 **Diagnostiquer** la présence de la maladie de Parkinson (classification binaire)
- 📊 **Évaluer la sévérité** via les scores UPDRS (régression)
- 🎯 **Fournir un outil clinique complet** en un seul modèle unifié

### **Secondaires**
- ✅ Implémenter un réseau de neurones **from scratch** en Rust
- ✅ Créer une architecture **modulaire et extensible**
- ✅ Optimiser les performances pour un usage **temps réel**
- ✅ Assurer la **robustesse** nécessaire au domaine médical

## 🏗️ Architecture Technique

### **Structure du Modèle**
```
[Input: 22 features] → [Shared Encoder] → [Features Partagées]
                                  ↓
              [Branche Classification]   [Branche Régression]
                          ↓                         ↓
                   [Status: 0/1]             [UPDRS Score]
```

### **Composants Clés**
- **Shared Encoder** : Couches communes apprenant les patterns vocaux
- **Tête Classification** : Spécialisée dans le diagnostic binaire
- **Tête Régression** : Spécialisée dans l'évaluation de sévérité
- **Système d'Alternance** : Entraînement intelligent multi-datasets

## 📊 Datasets Utilisés

### **1. Dataset de Classification** (Oxford)
- **📏 Taille** : 197 enregistrements vocaux
- **🎯 Cible** : `status` (0 = sain, 1 = Parkinson)
- **👥 Patients** : 31 personnes (23 Parkinson, 8 sains)
- **📋 Features** : 22 mesures vocales (fréquence, amplitude, bruit, complexité)

### **2. Dataset de Télémonitoring** (Oxford)
- **📏 Taille** : 5,875 enregistrements
- **🎯 Cibles** : `motor_UPDRS` et `total_UPDRS` (scores de sévérité)
- **👥 Patients** : 42 patients Parkinson
- **📋 Features** : 16 mesures vocales + données temporelles

## 🚀 Fonctionnalités Implémentées

### **Noyau Réseau de Neurones**
- ✅ **MLP Complet** avec couches dense, activation, propagation
- ✅ **Backpropagation** efficace avec calcul de gradients
- ✅ **Optimiseurs** : SGD avec momentum, Adam
- ✅ **Fonctions d'activation** : ReLU, Sigmoid, Tanh, Linear, Softmax

### **Système d'Entraînement Avancé**
- ✅ **Alternance intelligente** entre datasets
- ✅ **Learning Rate Adaptatif** avec scheduling
- ✅ **Gradient Clipping** pour la stabilité
- ✅ **Early Stopping** automatique
- ✅ **Monitoring complet** des métriques

### **Gestion des Données**
- ✅ **Chargement CSV** des datasets réels
- ✅ **Normalisation automatique** des features
- ✅ **Shuffling** et batching intelligent
- ✅ **Validation croisée** par patient

## ⚡ Difficultés Rencontrées et Solutions

### **🎯 Défi 1 : Dimensions Incompatibles**
**Problème** : Les deux datasets ont des features différentes (22 vs 16)
**Solution** : Architecture modulaire avec encoder partagé + têtes spécialisées

### **🎯 Défi 2 : Échelles de Features Variables**
**Problème** : Jitter (0.001-0.02) vs HNR (10-30) → échelles très différentes
**Solution** : Normalisation robuste + initialisation intelligente des poids

### **🎯 Défi 3 : Apprentissage Multi-Tâches**
**Problème** : Les gradients des deux tâches s'interfèrent
**Solution** : Alternance stratégique + pondération des losses

### **🎯 Défi 4 : Données Médicales Limitées**
**Problème** : 197 samples pour la classification
**Solution** : Transfer learning depuis le dataset régression (5,875 samples)

### **🎯 Défi 5 : Performance Rust/NDArray**
**Problème** : Gestion manuelle de la mémoire et dimensions
**Solution** : Utilisation intensive de views() + opérations batch

## 📈 Résultats et Performances

### **Métriques Cibles**
- **Classification** : > 85% accuracy (diagnostic)
- **Régression** : RMSE < 3 points UPDRS (sévérité)
- **Temps d'inférence** : < 1ms par échantillon

### **Avancées Techniques**
- 🚀 **30x plus de données** via l'apprentissage multi-datasets
- 🎯 **Modèle unifié** plus robuste que deux modèles séparés
- ⚡ **Optimisation Rust** pour déploiement embarqué

## 🏥 Impact Médical et Applications

### **Usage Clinique**
```
Nouveau patient → [Modèle] → 
   📊 Probabilité Parkinson: 87%
   📈 Score UPDRS prédit: 28.4
   🎯 Diagnostic: "Risque élevé, sévérité modérée"
```

### **Applications Concrètes**
- 🏥 **Dépistage précoce** dans les centres de santé
- 📱 **Télémonitoring** à domicile des patients
- 🔬 **Recherche médicale** sur la progression de la maladie
- 🎓 **Outil éducatif** pour les professionnels de santé

## 🔬 Aspects Techniques Avancés

### **Innovations Architecturales**
- **Shared Encoder** : Apprentissage transfert entre tâches
- **Alternance Dynamique** : Ratio adaptatif pendant l'entraînement
- **Gradient Analysis** : Monitoring des gradients pour la stabilité

### **Optimisations Rust**
- **Memory Safety** : Pas de fuites mémoires, accès sécurisés
- **Performance** : Utilisation de NDArray pour le calcul scientifique
- **Parallelization** : Préparation pour le calcul parallèle

## 🛠️ Structure du Code

```
src/
├── main.rs                 # Point d'entrée et tests
├── neural_network/
│   ├── mod.rs             # Exports des modules
│   ├── layer.rs           # Implémentation des couches
│   ├── activation.rs      # Fonctions d'activation
│   ├── network.rs         # Réseau neuronal principal
│   └── optimizers.rs      # Algorithmes d'optimisation
```

## 🚀 Utilisation

### **Entraînement**
```rust
let mut network = NeuralNetwork::new(0.001);
network.add_layer(22, 64, Activation::Relu)
       .add_layer(64, 32, Activation::Relu)
       .add_layer(32, 1, Activation::Sigmoid);

let metrics = network.train(&inputs, &targets, 200, 32);
```

### **Inférence**
```rust
let prediction = network.forward(&patient_features);
println!("Risque Parkinson: {:.1}%", prediction[0] * 100.0);
```

## 📊 Métriques de Validation

### **Tests de Robustesse**
- ✅ **Cohérence** : Prédictions stables sur variations mineures
- ✅ **Généralisation** : Performance sur données invisibles
- ✅ **Calibration** : Scores bien calibrés médicalement
- ✅ **Temps réel** : Inférence < 1ms

## 🔮 Futures Améliorations

### **Court Terme**
- [ ] Intégration des données temporelles (séries chronologiques)
- [ ] Ajout de l'incertitude des prédictions
- [ ] Interface web pour démonstration

### **Long Terme**
- [ ] Modèle transformer pour séquences vocales
- [ ] Apprentissage fédéré pour la privacy
- [ ] Déploiement sur edge devices

## 🤝 Contribution

Ce projet est ouvert aux contributions, particulièrement :
- 🏥 **Experts médicaux** : Validation clinique et cas d'usage
- 🔬 **Data Scientists** : Amélioration des algorithmes
- 🦀 **Développeurs Rust** : Optimisation des performances

## 📚 Références

1. *Max A. Little, et al.* - "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease"
2. *Athanasios Tsanas, et al.* - "Accurate telemonitoring of Parkinson's disease progression"
3. *IEEE Transactions on Biomedical Engineering* - Publications originales

## ⚠️ Avertissements

- 🔬 **Recherche Expérimentale** : Ce projet est à but éducatif et de recherche
- 🏥 **Non Clinique** : Ne pas utiliser pour des diagnostics réels sans validation médicale
- 📊 **Données Simulées** : Les performances réelles peuvent varier

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

**💡 Innovation** : Premier réseau de neurones multi-tâches Parkinson implémenté en Rust avec architecture d'alternance intelligente.

**🎯 Impact** : Potentiel de révolutionner le diagnostic et suivi de la maladie de Parkinson via l'IA accessible.

**🚀 Future** : Base solide pour le développement d'outils médicaux IA open-source performants et sûrs.
