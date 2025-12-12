DR Score: Deep-Learning Radiomics for Knee OA Progression

This repository provides the official implementation of the DR Score, a continuous imaging biomarker derived from knee radiographs for predicting knee osteoarthritis progression and time-to-knee-replacement.

The method uses automated knee patch extraction and a Spatial–Representational kNN Attention MIL model to generate a 0–4 DR Score that reflects structural severity and future risk.

Features
Automated DR Score from a single X-ray
Attention-based MIL with survival prediction
Training pipeline (MOST + OAI)
External evaluation (MenTOR, KICK)
Simple daily-use inference script