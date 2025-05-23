# Performance-Fatigue-monitoring-using-EMGs

Performance Fatigue Analysis in Manual Handling Tasks
Performance fatigue is a key contributor to work-related musculoskeletal disorders (WMSDs). This project explores the relationship between myoelectric manifestations of fatigue (MMF) and perceived exertion during prolonged manual handling tasks (MHT). Surface EMG and IMUs were used to monitor physiological and biomechanical changes, enabling both correlation analysis and fatigue stage classification using deep learning.

## Publication
A manuscript based on this project has been submitted for peer review:

## Study Overview
Participants
  - 8 healthy individuals

Data Acquisition
  - sEMG: Trigno, Delsys | Sampling Frequency: 1200 Hz
  - sEMG placements: Biceps Brachii, Carpi Radialis, Trapezius, Deltoideus Post, Erector Spinae Longissimus (L), Erector Spinae Iliocostalis (I), Rectus Femoris, Tibialis Anterior, Biceps Femoris, Lateral Gastrocnemius
  - IMUs: MTws, Xsens Technologies | Sampling Frequency: 100 Hz
  - IMU placements: Sternum, Sacrum, Upper Arm (R), Forearm (R), Thigh (R), Shank (R), Foot (R)

Signal Processing
- sEMG Preprocessing:
  - Bandpass filter: 10–500 Hz
  - Smoothing: 50 ms moving average
  - Normalization: relative to the first 5 lifting cycles (non-fatigued baseline)

Model & Analysis
Correlation Analysis: MMF indicators were correlated with RPE values using Spearman's ρ
Fatigue Classification:
  - A deep neural network trained on extracted indicators
  - Achieved ~69% classification accuracy across 5 fatigue levels
