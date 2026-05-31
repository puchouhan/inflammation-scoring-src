# HPO Integration Guide

## Table of Contents
- [Überblick](#überblick)
- [HPO Modi](#hpo-modi)
  - [1. use_existing (Empfohlen für normale Nutzung)](#1-use_existing-empfohlen-für-normale-nutzung)
  - [2. overwrite (Für Neuoptimierung)](#2-overwrite-für-neuoptimierung)
  - [3. skip (Nur Default-Parameter)](#3-skip-nur-default-parameter)
- [Detaillierte Ausgaben](#detaillierte-ausgaben)
- [Datenfluss](#datenfluss)
- [Konfiguration](#konfiguration)
- [Verwendung](#verwendung)
  - [Szenario 1: Erste HPO für neue Modelle](#szenario-1-erste-hpo-für-neue-modelle)
  - [Szenario 2: Neues Modell zu existierenden hinzufügen](#szenario-2-neues-modell-zu-existierenden-hinzufügen)
  - [Szenario 3: Search Space geändert, Neuoptimierung nötig](#szenario-3-search-space-geändert-neuoptimierung-nötig)
  - [Szenario 4: Schnelles Testen ohne HPO](#szenario-4-schnelles-testen-ohne-hpo)
- [Gespeicherte Dateien](#gespeicherte-dateien)
- [Report-Integration](#report-integration)
- [Automatisches Laden](#automatisches-laden)
- [Datenbank-Struktur & Study-Verwaltung](#datenbank-struktur--study-verwaltung)
- [Modi-Vergleich: Entscheidungshilfe](#modi-vergleich-entscheidungshilfe)
- [Entscheidungsbaum](#entscheidungsbaum)
- [Troubleshooting](#troubleshooting)
- [Zusammenfassung](#zusammenfassung)
- [Command-Line Interface (CLI)](#command-line-interface-cli)

---

## Überblick

Das HPO-System ist vollständig in die Experiment-Pipeline integriert mit intelligenter Verwaltung von existierenden HPO-Ergebnissen.

## HPO Modi

Das System unterstützt drei Modi über den Parameter `hpo.mode`:

### 1. `use_existing` (Empfohlen für normale Nutzung)

```yaml
hpo:
  mode: "use_existing"
```

**Verhalten:**
- [DONE] Modelle MIT existierenden HPO-Ergebnissen: Verwendet optimierte Parameter, HPO wird übersprungen
- SEARCH: Modelle OHNE HPO-Ergebnisse: Führt HPO aus, um optimierte Parameter zu erzeugen
- TIP: Effizient: Optimiert nur was fehlt, spart Zeit

**Wann verwenden:**
- Normale Workflow: Einmal HPO, dann mehrere Training-Runs
- Neue Modelle zu bestehenden hinzufügen
- Ressourcen-effizient arbeiten

### 2. `overwrite` (Für Neuoptimierung)

```yaml
hpo:
  mode: "overwrite"
```

**Verhalten:**
- [RELOAD] Führt HPO für ALLE Modelle aus
- WARNING: Überschreibt existierende HPO-Ergebnisse
- TIME: Zeitaufwendig, aber garantiert frische Optimierung

**Wann verwenden:**
- Search space wurde geändert
- Neue/bessere HPO-Strategie testen
- Alte Ergebnisse nicht mehr vertrauenswürdig
- Finale Thesis-Experimente (komplette Neuoptimierung)

### 3. `skip` (Nur Default-Parameter)

```yaml
hpo:
  mode: "skip"
```

**Verhalten:**
- [SKIP] Keine HPO wird ausgeführt
- Training verwendet Parameter aus `configs/base.yaml` und `configs/models/*.yaml`
- INFO: Falls HPO-Ergebnisse existieren, werden diese beim Training trotzdem geladen (außer `use_hpo_params=False`)

**Wann verwenden:**
- Debugging/Schnelle Tests
- Baseline-Vergleich mit Default-Parametern
- HPO bereits extern durchgeführt
- Limitierte Rechenzeit
- **From-scratch-Modelle ohne TIMM-Backbone** (SimCLR, DINO, GNN) -- siehe unten

---

## Nicht-kompatible Modelle: SimCLR, DINO, GNN

Für die Modelle `simclr`, `dino` und `gnn` ist HPO **grundsätzlich nicht anwendbar**.
Notebook-Config für diese Modelle: `ENABLE_HPO = False`, `HPO_MODE = "skip"`.

**Grund 1: Kein Modell-YAML vorhanden**

Die HPO-Pipeline lädt via `load_config(model_name)` eine Datei
`configs/models/<model_name>.yaml`. Für SimCLR, DINO und GNN existiert diese
Datei nicht, da deren Hyperparameter fest im Modellcode verankert sind.
Ohne passendes YAML fällt `load_config` auf `base.yaml` zurück -- ein
undefinierter Zustand, der zu falschen Suchräumen führt.

**Grund 2: Inkompatibiler Suchraum**

Der Optuna-Suchraum in `src/hpo/hpo.py` (Zeilen ~50-120) wurde für
TIMM-supervisierte Backbones entwickelt. Die optimierten Parameter:

| Parameter | SimCLR | DINO | GNN |
|---|---|---|---|
| `learning_rate`, `weight_decay` | Marginal relevant | Marginal relevant | Marginal relevant |
| TIMM-Backbone-Dropout | Nein | Nein | Nein (kein TIMM) |
| Head-Typ, Feature-Dim | Nein | Nein | Nein |
| Scheduler-Patience | Eingeschränkt | Eingeschränkt | Eingeschränkt |

Die architektur-relevanten Hyperparameter dieser Modelle fehlen im Suchraum:
- SimCLR: Projektionskopf-Größe, Temperaturparameter tau
- DINO: EMA-Momentum (Teacher-Update-Rate), Centering-Rate
- GNN: Anzahl Graph-Layer, Hidden-Channels, Graph-Konstruktionsstrategie

**Grund 3: Kein pretrained-Pfad**

SimCLR und DINO nutzen `pretrained=False` (hardcoded in `ssl_model.py:14`
bzw. `dino_model.py:37,55`). GNN hat kein TIMM-Backbone. Der HPO-Suchraum
ist konzeptionell auf das Fine-Tuning vortrainierter Gewichte ausgelegt.

**Auswirkung auf den wissenschaftlichen Vergleich:**

Gruppe A (supervised TIMM) trainiert mit HPO-optimierten Parametern.
Gruppe B (SimCLR/DINO/GNN) trainiert mit `base.yaml`-Standardwerten.
Dieser Unterschied im Optimierungsgrad ist eine methodisch bekannte Limitation
und muss in der Thesis-Diskussion explizit erwähnt werden.
(Vollständige Formulierungsvorlage: `docs/THESIS_COMPARISON_PROTOCOL_2026-03-15.md` Abschnitt 9.3)

## Detaillierte Ausgaben

Das System gibt bei Ausführung genau aus, was mit jedem Modell passiert:

### Beispiel: `use_existing` Mode

```
================================================================================
HYPERPARAMETER OPTIMIZATION CONFIGURATION
================================================================================
Mode: use_existing

📋 USE_EXISTING Mode:
  - Models WITH existing HPO results: Use optimized parameters (skip HPO)
  - Models WITHOUT HPO results: Run HPO to generate optimized parameters
  - Efficient: Only optimizes what's missing

Search Mode: model_specific
Trials per model: 50
Folds per trial: 3
Target models: vit, efficientnetv2, swin

Status of HPO results:
  [DONE] vit                 - HPO results exist
  [FAILED] efficientnetv2      - No HPO results found
  [FAILED] swin                - No HPO results found

Models needing HPO: efficientnetv2, swin
TIME:  Estimated time: 7.5 - 12.5 hours
================================================================================

================================================================================
STARTING MODEL-SPECIFIC HPO
================================================================================

================================================================================
Model: VIT
================================================================================
[DONE] HPO results already exist for vit
   Skipping HPO (using existing optimized parameters)
   File: configs/hpo_best_*vit*.yaml
================================================================================

================================================================================
Model: EFFICIENTNETV2
================================================================================
[FAILED] No HPO results found for efficientnetv2
   Running HPO to generate optimized parameters...

SEARCH: Starting HPO for efficientnetv2...
   Trials: 50
   Folds: 3
   This may take a while...

[... HPO progress ...]

[DONE] HPO completed for efficientnetv2!
   Best QWK: 0.7234
   Results saved to: configs/hpo_best_hpo_efficientnetv2_master_run.yaml
================================================================================

[... same for swin ...]

================================================================================
HPO EXECUTION SUMMARY
================================================================================
       Model                    Status  Best QWK Trials   Best LR Best Batch
         vit  [SKIP] Skipped (using existing)         -      -         -          -
efficientnetv2             [DONE] Completed    0.7234     50  1.80e-04         64
        swin             [DONE] Completed    0.6891     50  2.30e-04         32
================================================================================

 HPO results location: configs/hpo_best_*.yaml
 View detailed results: optuna-dashboard sqlite:///optuna_study.db
```

### Beispiel: `overwrite` Mode

```
================================================================================
Mode: overwrite

[RELOAD] OVERWRITE Mode:
  - Run HPO for ALL models
  - Overwrite any existing HPO results
  - Use this when: search space changed, want fresh optimization

Status of HPO results:
  [DONE] vit                 - HPO results exist
  [DONE] efficientnetv2      - HPO results exist
  [FAILED] swin                - No HPO results found
================================================================================

================================================================================
Model: VIT
================================================================================
[RELOAD] HPO results exist for vit, but OVERWRITE mode is active
   Running HPO and overwriting existing results...

SEARCH: Starting HPO for vit...
[...]
```

### Beispiel: `skip` Mode

```
================================================================================
Mode: skip

[SKIP] SKIP Mode:
  - No HPO will be executed
  - Training will use default parameters from configs/
  - Existing HPO results will be loaded during training if available
================================================================================

[SKIP] HPO Mode is 'skip' - No hyperparameter optimization will be performed.
   Training will use default parameters from config files.
   (Existing HPO results will still be loaded during training if available)

Proceeding to training phase...
```

## Datenfluss

```

  1. HPO Phase   

         
          Optuna Study (SQLite DB)
            - Alle Trials
            - Beste Parameter
            - Study History
         
          configs/hpo_best_*.yaml
             - Beste Hyperparameter pro Modell
             - Wird automatisch beim Training geladen


 2. Training     

         
          Lädt HPO-Parameter (falls vorhanden)
            - learning_rate
            - weight_decay
            - batch_size
            - beta1, beta2
            - scheduler_patience
            - drop_rate
         
          Experiment Tracker
            - experiments/RUN_ID/MODEL_NAME/
            - config.yaml (inkl. HPO-Parameter)
            - metrics/
            - checkpoints/
         
          Report Generator
             - Liest HPO-Parameter aus config.yaml
             - Zeigt optimierte vs. default Parameter
```

## Konfiguration

### In base.yaml

```yaml
hpo:
  # HPO Mode - controls when HPO is executed
  # Options:
  #   "use_existing" - Use existing HPO results if available, run HPO only for models without results
  #   "overwrite"    - Run HPO for all models, overwrite existing results
  #   "skip"         - Skip HPO completely, use default parameters only
  mode: "use_existing"
  
  search_mode: "model_specific"  # "model_specific" or "multi_model"
  n_trials: 50
  n_folds: 3
  storage: "sqlite:///optuna_study.db"
  
  # Models to optimize (for model_specific search_mode)
  # If empty, will use all models from models_to_train
  models: []  # Example: ["vit", "efficientnetv2"]
  
  search_space:
    learning_rate:
      min: 1e-5
      max: 5e-4
      log: true
    # ... weitere Parameter
```

### Im Notebook

Die Parameter werden automatisch aus `base.yaml` geladen:

```python
# Alte Version ([NO] entfernt):
RUN_HPO = False
HPO_MODE = "model_specific"
HPO_TRIALS = 50
HPO_FOLDS = 3
HPO_MODELS = ["vit", "efficientnetv2"]

# Neue Version ([DONE] aus config):
RUN_HPO = config['hpo']['enabled']
HPO_MODE = config['hpo']['mode']
HPO_TRIALS = config['hpo']['n_trials']
HPO_FOLDS = config['hpo']['n_folds']
HPO_MODELS = config['hpo']['models'] or config['models_to_train']
```

## Verwendung

### Szenario 1: Erste HPO für neue Modelle

```yaml
# configs/base.yaml
hpo:
  mode: "use_existing"  # Empfohlen
  search_mode: "model_specific"
  n_trials: 50
  models: []  # Leer = alle models_to_train

models_to_train:
  - vit
  - efficientnetv2
  - swin
```

**Ergebnis:**
- Alle 3 Modelle haben keine HPO-Ergebnisse
- HPO wird für alle 3 ausgeführt
- Ergebnisse gespeichert in `configs/hpo_best_*.yaml`
- TIME: ~12-20 Stunden

### Szenario 2: Neues Modell zu existierenden hinzufügen

```yaml
hpo:
  mode: "use_existing"
  
models_to_train:
  - vit              # Hat bereits HPO-Ergebnisse
  - efficientnetv2   # Hat bereits HPO-Ergebnisse
  - swin             # Hat bereits HPO-Ergebnisse
  - regnety          # NEU - keine HPO-Ergebnisse
```

**Ergebnis:**
- vit, efficientnetv2, swin: HPO übersprungen (existierende Parameter verwendet)
- regnety: HPO wird ausgeführt
- TIME: ~4-7 Stunden (nur für regnety)

### Szenario 3: Search Space geändert, Neuoptimierung nötig

```yaml
hpo:
  mode: "overwrite"  # Alle neu optimieren
  models: ["vit", "efficientnetv2"]  # Nur diese beiden
  
  search_space:
    learning_rate:
      min: 1e-6  # Vorher: 1e-5 (größerer Bereich!)
      max: 1e-3  # Vorher: 5e-4
```

**Ergebnis:**
- vit: HPO wird neu ausgeführt (alte Ergebnisse überschrieben)
- efficientnetv2: HPO wird neu ausgeführt
- swin: Nicht in `models` Liste  wird übersprungen
- TIME: ~8-14 Stunden (für vit + efficientnetv2)

### Szenario 4: Schnelles Testen ohne HPO

```yaml
hpo:
  mode: "skip"  # Keine HPO
```

**Ergebnis:**
- Keine HPO wird ausgeführt
- Training verwendet Default-Parameter
- Falls HPO-Ergebnisse existieren, werden diese beim Training trotzdem geladen
- TIME: 0 Stunden HPO

## Gespeicherte Dateien

### Nach HPO

```
configs/
 hpo_best_hpo_vit_v2.yaml              # Beste vit-Parameter
 hpo_best_hpo_efficientnetv2_v2.yaml   # Beste efficientnetv2-Parameter
 ...

optuna_study.db                           # SQLite mit allen Trials
```

Beispiel `hpo_best_hpo_vit_v2.yaml`:
```yaml
learning_rate: 0.00023456
weight_decay: 2.3e-05
beta1: 0.92
beta2: 0.997
scheduler_patience: 5
batch_size: 32
drop_rate: 0.15
```

### Nach Training

```
experiments/2026-01-05_14-30-00/
 vit/
    config.yaml          # [DONE] Enthält HPO-Parameter (falls verwendet)
    metrics/
       fold_0_metrics.json
       model_complexity.json
    checkpoints/
    tensorboard/
 efficientnetv2/
    config.yaml          # [DONE] Enthält HPO-Parameter (falls verwendet)
    ...
 report.pdf               # Zeigt HPO vs Default Parameters
```

## Report-Integration

Der Report Generator zeigt automatisch:

1. **Verwendete Hyperparameter**: Default vs. HPO-optimiert
2. **HPO-Historie**: Convergence plots (falls vorhanden)
3. **Parameter Importance**: Welche Parameter am wichtigsten waren
4. **Vergleich**: Modelle mit/ohne HPO

## Automatisches Laden

Das Training lädt HPO-Parameter automatisch:

```python
# In train_model():
def train_model(model_name, config, run_id, fold_idx=0, use_hpo_params=True):
    if use_hpo_params:
        # Lädt configs/hpo_best_*.yaml wenn vorhanden
        config = merge_hpo_config(config, model_name)
        # [DONE] Verwendet optimierte Parameter
    
    # Training mit (optimierter oder default) config
    model = ModelFactory.create_model(model_name, config)
    ...
```

Um HPO-Parameter NICHT zu verwenden:

```python
metrics = train_model(model_name, config, run_id, use_hpo_params=False)
```

## Datenbank-Struktur & Study-Verwaltung

Die SQLite-Datenbank (`optuna_study.db`) enthält alle HPO-Studies:

```
Studies in optuna_study.db:
 hpo_vit_master_run           # Study für vit
    Trial 0: QWK=0.65, params={lr=1e-4, ...}
    Trial 1: QWK=0.68, params={lr=2e-4, ...}
    ...
    Trial 49: QWK=0.72, params={lr=1.5e-4, ...}

 hpo_efficientnetv2_master_run
    50 Trials...

 hpo_multimodel_master_run    # Multi-Model Study
     100 Trials...
```

### Study-Verwaltung je nach Modus

#### `use_existing` Mode:
```python
# Bestehende Study wird NICHT verändert
study = optuna.create_study(
    study_name="hpo_vit_master_run",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True  # Lädt existierende Study
)
# WARNING: Kein optimize() Aufruf  0 neue Trials
```

**Ergebnis:**
- Alte Study bleibt intakt (z.B. 50 Trials)
- Keine neuen Trials werden hinzugefügt
- YAML-File wird nicht überschrieben
- Effizient: Keine Rechenzeit verschwendet

#### `overwrite` Mode:
```python
# Alte Study wird GELÖSCHT, neue erstellt
optuna.delete_study(
    study_name="hpo_vit_master_run",
    storage="sqlite:///optuna_study.db"
)
# Alte Trials (50) gelöscht [NO]

study = optuna.create_study(
    study_name="hpo_vit_master_run",
    storage="sqlite:///optuna_study.db",
    load_if_exists=False  # Erstellt neue Study
)
study.optimize(objective, n_trials=50)
# Neue Trials (50) hinzugefügt [DONE]
```

**Ergebnis:**
- Alte Study komplett gelöscht
- Frische Study mit 50 neuen Trials
- YAML-File überschrieben mit neuen Best-Parametern
- Clean Slate: Alte Ergebnisse verworfen

### Warum KEINE "Fortsetzen"-Option?

**Problem mit Fortsetzen:**
```python
# Session 1: 50 Trials mit alter Config
study.optimize(objective, n_trials=50)  # search_space: lr [1e-5, 5e-4]

# Session 2: 30 Trials mit NEUER Config
# search_space: lr [1e-6, 1e-3]  # Größerer Bereich!
study.optimize(objective, n_trials=30)  # Total 80 Trials
```

**Problem:** Trials 0-49 und Trials 50-79 haben unterschiedliche Search Spaces!
- Optuna's TPE Sampler lernt von allen Trials
- Aber frühe Trials haben engeren Bereich  verzerrte Optimierung

**Unsere Lösung:**
- `use_existing`: Nutze alte 50, füge 0 hinzu  konsistent
- `overwrite`: Lösche alte 50, erstelle neue 50  konsistent
- Keine Vermischung  saubere Ergebnisse

### Search Space Validierung

**Frage:** Soll geprüft werden ob sich search_space geändert hat?

**Antwort:** NEIN - User-Verantwortung

**Begründung:**
1. **Komplexität:** Tiefes Dict-Vergleichen ist fehleranfällig
2. **Flexibilität:** User will vielleicht bewusst alte Ergebnisse trotz Änderung nutzen
3. **Einfachheit:** Klare Regel: "Config geändert?  `mode: overwrite`"

**Best Practice:**
```yaml
# Wenn du search_space änderst:
hpo:
  mode: "overwrite"  # Explizit neu optimieren
```

**Dokumentation in Thesis:**
```latex
To ensure consistency, we re-ran HPO with mode='overwrite' 
after modifying the search space, discarding all previous 
optimization results.
```

## Modi-Vergleich: Entscheidungshilfe

| Situation | Empfohlener Modus | Begründung |
|-----------|------------------|------------|
| 🆕 Erste HPO für alle Modelle | `use_existing` | Alle werden optimiert (keine existieren) |
|  Ein neues Modell hinzufügen | `use_existing` | Nur das neue wird optimiert |
|  Search space geändert | `overwrite` | Alte Ergebnisse nicht mehr passend |
| 🎓 Finale Thesis-Experimente | `overwrite` | Garantiert frische, konsistente Optimierung |
| 🐛 Debugging/Quick Test | `skip` | Spart Zeit, verwendet Defaults |
|  Baseline-Vergleich | `skip` | Will explizit Default-Parameter testen |
| 💾 Rechenzeit sparen | `use_existing` | Nutzt vorhandene Arbeit |
| [RELOAD] Zweiter Training-Run | `skip` oder `use_existing` | Erste HPO bereits durchgeführt |

## Entscheidungsbaum

```
Brauche ich HPO?

 Nein, will nur Default-Parameter testen
   mode: "skip"

 Ja, will optimierte Parameter
   
    Existieren bereits HPO-Ergebnisse?
     
      Nein
        mode: "use_existing" (führt HPO aus)
     
      Ja
        
         Sind die Ergebnisse noch aktuell?
          (Search space unverändert, gleiche Daten, etc.)
          
           Ja
             mode: "use_existing" (überspringt HPO, nutzt alte)
          
           Nein (Config geändert, will neu optimieren)
              mode: "overwrite" (führt HPO neu aus)
        
         Will ich trotz guter Ergebnisse neu optimieren?
           (z.B. finale Thesis-Experimente, bessere Methode)
            mode: "overwrite"
```

### 1. Wann HPO aktivieren?

[DONE] **Aktivieren für**:
- Finale Thesis-Experimente (optimale Performance)
- Neue Modell-Architekturen (unbekannter optimaler Bereich)
- Wenn Baseline-Ergebnisse nicht zufriedenstellend sind

[SKIP] **Überspringen für**:
- Quick Debugging/Testing
- Wiederholte Runs mit bekannten guten Parametern
- Zeitlich begrenzte Ressourcen

### 2. HPO-Parameter in Thesis dokumentieren

```latex
\subsection{Hyperparameter Optimization}

We used Bayesian optimization (TPE sampler) to find optimal 
hyperparameters for each architecture. The search space included:

\begin{itemize}
    \item Learning rate: $[10^{-5}, 5 \times 10^{-4}]$ (log-scale)
    \item Weight decay: $[10^{-6}, 10^{-3}]$ (log-scale)
    \item Batch size: $\{16, 32, 64\}$
    \item Dropout rate: $[0.0, 0.3]$
    \item Optimizer momentum ($\beta_1$, $\beta_2$)
    \item Scheduler patience: $\{3, 4, 5, 6, 7\}$ epochs
\end{itemize}

For each architecture, we ran 50 trials with 3-fold 
cross-validation per trial. The best parameters were:

\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
Model & LR & Weight Decay & Batch Size & Dropout \\
\midrule
ViT & $2.3 \times 10^{-4}$ & $2.3 \times 10^{-5}$ & 32 & 0.15 \\
EfficientNetV2 & $1.8 \times 10^{-4}$ & $1.5 \times 10^{-5}$ & 64 & 0.20 \\
\bottomrule
\end{tabular}
\caption{HPO-optimized hyperparameters per architecture}
\end{table}
```

### 3. Reproducibility

Alle HPO-Ergebnisse sind reproduzierbar:

```python
# In base.yaml
seed: 42  # Fester Seed für alle Experimente

# In hpo.py
sampler = optuna.samplers.TPESampler(seed=42)  # Deterministisch
```

Die SQLite-Datenbank und YAML-Files ermöglichen:
- [DONE] Nachvollziehbarkeit aller HPO-Entscheidungen
- [DONE] Wiederholbarkeit mit exakt gleichen Parametern
- [DONE] Vergleichbarkeit verschiedener HPO-Runs

## Troubleshooting

### Problem: HPO-Parameter werden nicht geladen

**Check 1**: Existiert die YAML-Datei?
```bash
ls configs/hpo_best_*.yaml
```

**Check 2**: Richtiger Dateiname?
Die Funktion sucht nach:
- `hpo_best_hpo_{model}_v2.yaml`
- `hpo_best_hpo_{model}_master_run.yaml`
- `hpo_best_{model}.yaml`

**Fix**: Benenne die Datei um oder passe `load_hpo_config()` an.

### Problem: Training ignoriert HPO-Parameter

**Check**: `use_hpo_params=True` gesetzt?
```python
metrics = train_model(model_name, config, run_id, use_hpo_params=True)  # [DONE]
```

### Problem: Optuna DB corrupt

```bash
# Backup erstellen
cp optuna_study.db optuna_study.db.backup

# Neue Study starten
python -m src.hpo --model vit --trials 50 --folds 3
```

## Zusammenfassung

[DONE] **Vorteil des integrierten Systems**:

1. **Single Source of Truth**: Alle HPO-Settings in `base.yaml`
2. **Automatisches Laden**: Training verwendet HPO-Parameter ohne manuelle Anpassung
3. **Nachvollziehbarkeit**: Jedes Experiment speichert verwendete Parameter
4. **Reproducibility**: SQLite DB + YAML ermöglichen exakte Wiederholung
5. **Flexibilität**: HPO kann aktiviert/deaktiviert werden ohne Code-Änderungen

Das System dokumentiert automatisch, ob und welche HPO-Parameter verwendet wurden!

## Command-Line Interface (CLI)

### Basis-Verwendung

```bash
# Model-specific HPO (empfohlen)
python -m src.hpo --model vit --trials 50 --folds 3
python -m src.hpo --model efficientnetv2 --trials 50 --folds 3

# Warum model-specific?
# - Verschiedene Architekturen haben unterschiedliche Optimierungslandschaften
# - CNNs: höhere LR (1e-4), stärkeres weight decay
# - Transformers: kleinere LR (1e-5), sensibel auf warmup
# - Hybrid Models: ausbalancierte Parameter
# - Batch Size Abhängigkeiten: große Modelle brauchen kleinere Batches
# Resultat: Optimale Performance pro Architektur, faire Vergleichbarkeit

# Multi-model HPO (sucht über Architekturen)
python -m src.hpo --trials 100 --folds 3
```

### Mit `--overwrite` Flag

```bash
# Löscht existierende Study und startet neu
python -m src.hpo --model vit --trials 50 --folds 3 --overwrite

# Ausgabe:
# [RELOAD] Deleted existing study 'hpo_vit_v2' (overwrite mode)
# [DONE] Starting fresh optimization with 50 trials...
```

**Ohne `--overwrite`:**
- Lädt existierende Study
- Fügt neue Trials zu bestehenden hinzu
- WARNING: Nur sinnvoll wenn search_space unverändert!

**Mit `--overwrite`:**
- Löscht alte Study komplett
- Erstellt neue Study von Grund auf
- [DONE] Empfohlen nach Config-Änderungen

### Weitere Optionen

```bash
# Schneller Test (weniger Trials/Folds)
python -m src.hpo --model vit --trials 10 --folds 2 --overwrite

# Custom study name
python -m src.hpo --model vit --study-name my_vit_experiment_v3

# Kombiniert
python -m src.hpo --model swin --trials 30 --folds 3 --overwrite --study-name swin_final
```

### Optuna Dashboard

Visualisiere alle Studies:

```bash
# Dashboard starten
optuna-dashboard sqlite:///optuna_study.db

# Browser öffnet sich auf http://localhost:8080
```

**Features:**
-  Convergence plots (QWK über Trials)
- 📈 Parameter importance (welche Parameter matter)
-  Parallel coordinate plots
-  Trial history und Details
- SEARCH: Filter nach Study name

**Nach `overwrite`:**
- Alte Study ist weg (gelöscht aus DB)
- Nur neue Study erscheint im Dashboard
