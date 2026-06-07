# Black-Box Modelling of Multiband Saturation

**Deep Learning Course · FCUL 2025/26**

## Overview

This project explores **black-box modelling of a multiband saturation VST plugin** ([FabFilter Saturn 2](https://www.fabfilter.com/products/saturn-2-multiband-distortion-saturation-plug-in)) applied to electric bass guitar, using raw waveform deep learning. Two architectures are trained and compared on clean and saturated audio file pairs:

- **Long Short-Term Memory (LSTM)** — Recurrent Neural Network
- **WaveNet** — Dilated Causal Convolutional Neural Network

Training uses a combined **ESR + DC + MRSTFT** loss (via [auraloss](https://github.com/csteinmetz1/auraloss)) to optimise time-domain accuracy and spectral fidelity.

## Dataset

This project uses the **IDMT-SMT-Bass** dataset by Fraunhofer IDMT (~5,200 direct input electric bass WAV files).

- Access the dataset at: https://www.idmt.fraunhofer.de/en/publications/datasets/bass.html

## Audio Demonstration

Audio examples comparing the trained models against the FabFilter Saturn 2 target are available at: [Black-Box Modelling of Multiband Saturation — Audio Examples](https://joao-canais.github.io/Black-Box-Modelling-of-Multiband-Saturation/)

## Abstract

Virtual analog modelling has become an active area of research as **musicians and producers seek software alternatives to expensive and inaccessible hardware processors**. Multiband saturators present a particularly demanding emulation target: they combine **frequency-dependent nonlinear distortion** across several bands with crossover filtering. This work addresses **black-box modelling of multiband saturation** using paired electric bass recordings from the IDMT-SMT-Bass dataset. We compare a **bidirectional LSTM** and a **WaveNet-style convolutional network** trained with a combined time-domain and multi-resolution spectral loss.
