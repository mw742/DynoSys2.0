# DynoSys 2.0: Graph-Based Modeling of Dynamic Risk States

DynoSys 2.0 extends the original DynoSys framework by introducing **graph-based and temporal modeling** of human behavioral development.

This framework represents each individual as a **dynamic system**, where interactions between genetic, environmental, and neurobiological domains evolve over time.

---

## 🧠 Key Idea

Human behavior is modeled as a **time-evolving graph**:

- **Nodes** → domain-level features  
  (genetics, environment, brain, etc.)

- **Edges** → relationships between domains  
  (statistical + temporal dependencies)

- **Time** → graph sequence per individual  

This enables:
- modeling **interactions across domains**
- capturing **temporal evolution**
- identifying **dynamic risk states**

---

## 🔁 From DynoSys → DynoSys 2.0

| Version | Representation | Limitation | Improvement |
|--------|--------------|-----------|------------|
| DynoSys (v1) | Tabular / domain summaries | Independent features | Predictive integration |
| DynoSys 2.0 | Graph + temporal system | — | Captures interactions + dynamics |

---

## 📊 Data

Input data could be prepared in **longitudinal long format**:

Each row:
- one individual  
- one time point  

All domains (genetic, environmental, brain, behavioral) must be aligned by:

- subject ID
- time index

---

## 🧩 Graph Construction

### Nodes
Represent domain-level features:
- PRS
- environmental domains
- brain representations

### Edges
Constructed using:
- statistical associations (feature selection)
- lagged relationships (temporal dependencies)
- optional sparsity thresholding

---

## ⏱ Temporal Graph Modeling

Each individual is represented as a sequence of graphs.

---

## 🤖 Modeling Approaches

### 1. Sequence Model (LSTM / GRU)
- Input: flattened node features
- Captures temporal dynamics

### 2. Graph-Temporal Model
- Input: graph structure + node features
- Methods:
  - Graph Neural Networks (GNN)
  - Recurrent layers (GRU / LSTM)

---

## 📈 System-Level Metrics

We define system-level properties:

### Graph Energy
- magnitude of node activity  
- connectivity-weighted signal  

### Stability
- variance across time  

### Transitions
- large changes between time points  

These metrics quantify **dynamic system behavior**.


## Data access and responsible use

This repository distributes analysis code and does not include participant-level research data. Users must obtain input data through the appropriate authorized providers and comply with applicable data-use agreements and institutional requirements.

For reproducibility, record the cohort, data release/version, accession or DOI where available, preprocessing steps, and exact code commit used.

Generated files may contain participant identifiers, individual measurements, outcomes, or predictions. Do not upload these files to a public repository without authorization. The software license does not grant permission to redistribute external datasets.

## Validation status and updates

This repository contains research code that may include undetected errors or limitations. The current source-code review did not include end-to-end execution or independent reproduction of the associated study results.

Known issues identified during review include import/path inconsistencies and a fallback that uses the test set for model selection when a separate validation set is unavailable. These issues require correction and validation.

Users should independently assess suitability for their analysis. Corrections following further checks will be documented in the commit history or release notes, including effects on outputs when known.


---

## Citation

If you use this framework, please cite:

Wei, M., & Peng, Q. (2026).
*DynoSys 2.0: Graph-Based Modeling of Dynamic Risk States and System Transitions in Human Behavioral Development.*
doi: https://doi.org/10.64898/2026.05.06.723259


