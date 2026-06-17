\# Audio Genre Classifier 🎵



A machine learning project to classify music genres using 

signal processing and classical ML, with CNN coming next.



\## Dataset

GTZAN Dataset — 1000 audio clips across 10 genres  

(blues, classical, country, disco, hiphop, jazz, metal, 

pop, reggae, rock), 30 seconds each.



\## Approach

\### Phase 1 — Signal Processing \& Feature Extraction

\- Visualized waveforms and spectrograms for all 10 genres

\- Extracted 41 features per song using librosa:

&#x20; - MFCC mean + std (26 features) — timbral texture

&#x20; - Chroma (12 features) — harmonic/pitch content

&#x20; - Spectral centroid — brightness of sound

&#x20; - Spectral rolloff — energy distribution

&#x20; - Zero crossing rate — percussiveness



\### Phase 2 — Classical ML Models

| Model | Accuracy |

|-------|----------|

| Random Forest (13 features) | 56% |

| Random Forest (41 features) | 66% |

| Random Forest (GridSearch tuned) | 64% |

| SVM RBF | 66% |

| SVM (GridSearch tuned) | 71% |



\### Phase 3 — CNN on Spectrograms (coming soon)



\## Key Insights

\- Classical music achieved highest accuracy (\~95%) — 

&#x20; distinct timbre separates it clearly from all other genres

\- Rock was hardest to classify — stylistically borrows 

&#x20; from metal, country and jazz making its feature 

&#x20; fingerprint inconsistent

\- SVM outperforms Random Forest on this dataset — 

&#x20; max-margin classifiers suit small high-dimensional data

\- Both untuned models plateaued at 66% — confirmed as 

&#x20; a feature ceiling, not a model limitation

\- Tuned SVM broke through to 71% with C=10, gamma=0.01



\## Signal Processing Insight

Adding MFCC standard deviation captured temporal dynamics.

Chroma features drove jazz improvement — jazz uses complex 

harmonic progressions distinct from other genres.

Zero crossing rate successfully differentiates percussive 

genres (metal, rock) from tonal ones (classical, jazz).



\## Tech Stack

Python, librosa, scikit-learn, matplotlib, seaborn



\## Spectrograms by Genre

!\[Spectrograms](spectrograms\_all\_genres.png)

