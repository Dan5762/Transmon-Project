# Transmon-Project

In this repository we have the code used to explore and model several possible circuit designs for a future charge tomography measurement. This code was developed for a MSc research project entitled "Simulating Superconducting Circuit for Charge-Basis Tomography".

## Getting Started

To get started clone the repository and install the required packages by running
```
pip install -r requirements.txt
```
from the root of the repository directory.

## Overview

The code is divided between circuit models and analytics, with the vast majority of focus put into circuit models. Circuit models come in four categories:
 - Directly Coupled Circuit
     - Probe-target qubits coupled via a capacitor
     - Longitudinal and transversal coupling is explored
     - Tunability of the qubits is explored
     - Impact of alpha dispersion (variability in probe qubit junctions) is explored
     - Bayesian optimisation of the circuit parameters is implemented
 - Indirectly (Resonator) Coupled Circuit
     - Probe-target qubits coupled via a resonator
     - Longitudinal and transversal coupling is explored
     - Impact of alpha dispersion (variability in probe qubit junctions) is explored
     - Impact of charge state cut-off is inspected
 - Flux Qubit
     - Simple model of a flux qubit
     - Qubit spectrum is plotted for a range of gate charges
 - C-Shunt Circuit
     - Single three junction capacitively shunted qubit
     - Limited time spent on this circuit so far

The circuit model notebooks typically have a system class which describes characterises the quantum system, they then use this class to inspect properties of the circuit in certain parameter regimes.

## Useful Papers

### Simulating Superconducitng Circuits Fundamentals
- [The superconducting circuit companion -- an introduction with worked examples<br/>S. E. Rasmussen, K. S. Christensen, S. P. Pedersen, L. B. Kristensen, T. Bækkegaard, N. J. S. Loft, N. T. Zinner](https://arxiv.org/abs/2103.01225)

- [A Quantum Engineer's Guide to Superconducting Qubits<br/>Philip Krantz, Morten Kjaergaard, Fei Yan, Terry P. Orlando, Simon Gustavsson, William D. Oliver](https://arxiv.org/abs/1904.06560)

### Project Specific Papers

- [Circuit-QED-based scalable architectures for quantum information processing with superconducting qubits<br/>P.-M. Billangeon, J. S. Tsai, and Y. Nakamura](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.91.094517)

- [Synthesizing arbitrary quantum states in a superconducting resonator<br/>Max Hofheinz, H. Wang, M. Ansmann, Radoslaw C. Bialczak, Erik Lucero, M. Neeley, A. D. O'Connell, D. Sank, J. Wenner, John M. Martinis & A. N. Cleland](https://www.nature.com/articles/nature08005)

- [Observation of quantum state collapse and revival due to the single-photon Kerr effect<br/>Gerhard Kirchmair, Brian Vlastakis, Zaki Leghtas, Simon E. Nigg, Hanhee Paik, Eran Ginossar, Mazyar Mirrahimi, Luigi Frunzio, S. M. Girvin & R. J. Schoelkopf](https://www.nature.com/articles/nature11902)