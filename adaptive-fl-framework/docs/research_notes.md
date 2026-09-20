# Research Notes — Adaptive Privacy-Aware Federated Learning

## 1. Existing Work Survey

### 1.1 Federated Learning with Differential Privacy

**Abadi et al. (2016) — "Deep Learning with Differential Privacy" (CCS 2016)**
- Introduced the moments accountant for tight DP accounting in SGD
- Per-sample gradient clipping + Gaussian noise (the foundation of Opacus)
- Fixed noise multiplier throughout training

**McMahan et al. (2018) — "Learning Differentially Private Recurrent Language Models"**
- Applied DP-SGD to federated learning
- Fixed epsilon target, fixed noise schedule
- Limitation: same noise applied regardless of training stage

**Geyer et al. (2017) — "Differentially Private Federated Learning: A Client Level Perspective"**
- Client-level DP (noise added to full model update, not per-sample gradients)
- Fixed clipping norm and noise multiplier
- Limitation: no adaptation to convergence state

### 1.2 Adaptive Noise / Privacy Scheduling

**Li et al. (2022) — "Auditing Differentially Private Machine Learning"**
- Empirical privacy auditing — measures actual epsilon vs theoretical
- Does not adapt noise during training

**Mironov et al. (2017) — Rényi Differential Privacy**
- Tighter privacy accounting via RDP
- Used by Opacus internally
- Still uses fixed noise schedule

**Andrew et al. (2021) — "Differentially Private Learning with Adaptive Clipping"**
- Adapts the CLIPPING NORM based on gradient quantiles
- Does NOT adapt the noise multiplier
- Does NOT use accuracy/convergence signals
- Limitation: only one parameter adapted, no utility feedback loop

**Noble et al. (2022) — "Differentially Private Federated Learning on Heterogeneous Data"**
- Addresses non-IID data with DP
- Fixed noise schedule
- Limitation: no runtime adaptation

**Xie et al. (2019) — "Local Differential Privacy for Deep Learning"**
- Local DP (noise added at client before sending)
- Fixed noise level
- Limitation: no adaptation

### 1.3 Privacy-Utility Tradeoff Optimisation

**Bagdasaryan et al. (2019) — "Differential Privacy Has Disparate Impact on Model Accuracy"**
- Shows DP disproportionately hurts minority classes
- Fixed DP parameters
- Limitation: no dynamic adjustment

**Tramèr & Boneh (2021) — "Differentially Private Learning Needs Better Features"**
- Shows that with better features, DP cost is lower
- Does not address adaptive scheduling

### 1.4 Convergence-Aware FL

**Li et al. (2020) — "Convergence of FedProx"**
- Analyses convergence under heterogeneous data
- No privacy component

**Karimireddy et al. (2020) — SCAFFOLD**
- Variance reduction in FL
- No privacy component

---

## 2. Identified Gap

### What exists:
- Fixed DP noise schedules (most work)
- Adaptive clipping norm only (Andrew et al. 2021)
- Privacy accounting without feedback (Opacus, RDP)
- Convergence analysis without privacy (FedProx, SCAFFOLD)

### What is MISSING:
**No existing work simultaneously:**
1. Monitors multiple runtime signals (accuracy, loss, convergence rate, privacy budget)
2. Uses those signals to jointly adapt BOTH noise multiplier AND clipping norm
3. Implements a documented, reproducible mathematical rule for the adaptation
4. Evaluates the resulting privacy-utility tradeoff against fixed-DP baselines
5. Integrates with Secure Aggregation

---

## 3. Our Proposed Contribution

### Title:
**Convergence-Aware Adaptive Differential Privacy Controller for Federated Learning**

### Core Idea:
At each FL round, the controller observes:
- Current test accuracy (utility signal)
- Accuracy improvement rate (convergence signal)
- Cumulative epsilon consumed (privacy budget signal)
- Training loss trend (stability signal)

It then adjusts the noise multiplier using a **proportional-integral rule**:

```
convergence_rate = (acc_t - acc_{t-1}) / max(acc_{t-1}, epsilon)

if convergence_rate > threshold_high:
    # Model is learning fast — can afford more privacy protection
    noise_multiplier *= (1 + alpha)   # increase noise

elif convergence_rate < threshold_low:
    # Model is converging slowly — reduce noise to help learning
    noise_multiplier *= (1 - beta)    # decrease noise

else:
    # Stable convergence — maintain current noise
    noise_multiplier unchanged

# Hard constraints:
noise_multiplier = clip(noise_multiplier, noise_min, noise_max)
```

Where:
- `alpha` = noise increase rate (e.g. 0.1 = 10% increase)
- `beta`  = noise decrease rate (e.g. 0.05 = 5% decrease)  
- `threshold_high` = convergence rate above which we increase privacy
- `threshold_low`  = convergence rate below which we decrease privacy
- `noise_min`, `noise_max` = hard bounds on noise multiplier

### Why this is defensible as novel:
1. **Multi-signal feedback**: uses both accuracy AND convergence rate (not just one)
2. **Asymmetric adaptation**: increase rate ≠ decrease rate (conservative privacy increase)
3. **Budget-aware**: slows noise reduction when cumulative epsilon is high
4. **Documented rule**: fully reproducible, deterministic given same inputs
5. **Evaluated against baselines**: compared to fixed-DP and no-DP

### What we do NOT claim:
- We do not claim this is the first adaptive DP system
- We do not claim cryptographic novelty
- We claim: a specific, implemented, evaluated controller that uses convergence
  signals to adapt noise, with documented tradeoffs

---

## 4. Experimental Plan

| Experiment | Config | Purpose |
|---|---|---|
| A | FedAvg, no DP | Accuracy ceiling |
| B | FedAvg + Fixed DP (noise=1.0) | DP baseline |
| C | FedAvg + Fixed DP + SecAgg+ | Security baseline |
| D | FedAvg + Adaptive DP + SecAgg+ | **Proposed method** |

Metrics compared:
- Final test accuracy
- Convergence speed (rounds to 30% accuracy)
- Privacy budget consumed (epsilon)
- Noise multiplier trajectory
- Training stability (loss variance)
