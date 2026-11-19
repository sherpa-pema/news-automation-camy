# Voice Reference Voice Audit & Selection Notes (Phase Q1)

## 1. Executive Summary
- **Original Baseline (`reference_v1.wav`) Vulnerabilities:**
  - Length: 4.89s (Too short; cause of VAE posterior sampling noise and acoustic drift)
  - Sample Rate: 24 kHz mono downsampled
  - F0 Stability: 18.7% drift across full run (204.2 Hz to 242.3 Hz), 3.6% on fixed 3-scene bench
  - Cosine Similarity Min: 0.8284 full run, 0.8454 on 3-scene bench

- **New Reference Deployed (`reference.wav`):**
  - Source: `config/voice/candidates/cand_3_cloned_long.wav`
  - Audio Spec: 16.7s, 44.1 kHz, 16-bit PCM mono, normalized to -3.0 dBFS peak
  - Cosine Similarity: **Min 0.9376** (+0.0922 vs baseline), **Std 0.0090** (lowest variance)
  - Pitch Stability: **F0 Spread 4.0%** (target <= 6.0%), Reference F0: 232.1 Hz
  - Cadence: **2.42 wps** (standard broadcast delivery), **WPS Variance 0.0205**
  - Persona: Preserves the established female anchor persona of the news bulletin while providing a clean 16.7s unclipped prompt.

---

## 2. Benchmark Candidate Comparison (3 Fixed Scenes: `intro`, `scene_01`, `scene_02`)

Synthesized and evaluated on NVIDIA GeForce GTX 1060 (6GB) using Ming-3.0-tts (bfloat16):

| Metric | Baseline (`reference_v1`) | Candidate 1 (`cand_1_emma_broadcast`) | Candidate 2 (`cand_2_emma_national`) | Candidate 3 (`cand_3_cloned_long`) [SELECTED] | Target |
|---|---|---|---|---|---|
| **Prompt Duration** | 4.89s | 14.4s | 16.2s | **16.7s** | 10–20s |
| **Prompt Sample Rate** | 24,000 Hz | 44,100 Hz | 44,100 Hz | **44,100 Hz** | 44.1/48 kHz |
| **Speaker Cosine Min** | 0.8454 | 0.8686 | 0.9006 | **0.9376** | >= 0.84 |
| **Speaker Cosine Std** | 0.0138 | 0.0192 | 0.0107 | **0.0090** | <= 0.02 |
| **F0 Spread %** | 3.6% (18.7% 8-sc) | 2.7% | 7.3% | **4.0%** | <= 6.0% (target <= 8%) |
| **F0 Range (Hz)** | 218.4 - 226.5 Hz | 190.9 - 196.0 Hz | 187.7 - 201.4 Hz | **214.1 - 222.7 Hz** | Consistent |
| **Scene LUFS Spread** | 0.4 LU | 0.4 LU | 0.2 LU | **0.3 LU** | <= 1.0 LU |
| **WPS Variance** | 0.0286 | 0.0240 | 0.0258 | **0.0205** | <= 0.08 |
| **Real-Time Factor (RTF)** | 3.85 | 3.95 | 3.86 | **3.87** | ~3.6-3.9 |

---

## 3. Candidate Transcripts

### Candidate 1 (`cand_1_emma_broadcast.wav`):
> "Good evening and welcome to the news briefing. Today we bring you comprehensive coverage of economic developments, regional trade, and community initiatives across the country. Stay tuned as we present the top stories and essential updates for tonight."

### Candidate 2 (`cand_2_emma_national.wav`):
> "Good evening, and welcome to tonight's national broadcast. In our lead reports, authorities in the capital confirmed updated infrastructure timelines and bilateral agreements. We bring you comprehensive reporting on economic policy, legal affairs, and regional transport developments."

### Candidate 3 (`cand_3_cloned_long.wav`) — DEPLOYED:
> "Good evening from the news desk. Tonight we review key national developments, including high-level government appointments, cross-border transport projects, and major parliamentary proceedings. Stay with us for verified reporting and detailed analysis from across the nation."

---

## 4. Rollback Instructions

If for any reason you wish to restore the original 4.89s reference voice:
```powershell
Copy-Item config\voice\reference_v1.wav config\voice\reference.wav -Force
Copy-Item config\voice\reference_v1.txt config\voice\reference.txt -Force
```
Candidate files remain archived under `config/voice/candidates/`.
