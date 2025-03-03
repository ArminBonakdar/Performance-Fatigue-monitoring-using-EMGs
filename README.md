# Performance-Fatigue-monitoring-using-EMGs

Performance Fatigue Analysis in Manual Handling Tasks
Performance fatigue significantly contributes to work-related musculoskeletal disorders, emphasizing the need to understand its effects during manual handling tasks. This study examines the correlation between changes in myoelectric manifestation of fatigue (MMF) indicators and participants' perceived exertion during prolonged manual handling tasks using surface electromyography (sEMG) sensors.

Methodology:
Ten sEMG sensors and seven inertial measurement units (IMUs) were used to assess MMF indicators and joint kinematics. sEMG recordings were segmented based on activity and joint range of motion. Linear and complexity-based MMF indicators were extracted and correlated with perceived exertion. A deep learning model classified fatigue stages with 69% accuracy.

Inputs:
- Number of participants: 8
- sEMG: Trigno, Delsys | Sampling Frequency: 1200 Hz
- sEMG placements: Biceps Brachii, Carpi Radialis, Trapezius, Deltoideus Post, Erector Spinae Longissimus (L), Erector Spinae Iliocostalis (I), Rectus Femoris, Tibialis Anterior, Biceps Femoris, Lateral Gastrocnemius
- IMUs: MTws, Xsens Technologies | Sampling Frequency: 100 Hz
- IMU placements: Sternum, Sacrum, Upper Arm (R), Forearm (R), Thigh (R), Shank (R), Foot (R)


Data preprocessing included bandpass filtering (10-500 Hz), smoothing (50 ms window), and normalization to the first five cycles (non-fatigue states). IMU-derived joint angles were computed following functional calibration.