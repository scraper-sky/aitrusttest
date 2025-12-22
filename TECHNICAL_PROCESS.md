# Technical Process & Architecture Diagram

## 🎯 Overall Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: DATA GENERATION                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   DatasetGenerator                   │
        │   - generate_math_items()            │
        │   - create_conversation()            │
        │   - generate_paired_dataset()        │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   List[Conversation]                 │
        │   - condition: "high_trust" |       │
        │              "low_trust"            │
        │   - turns: [user/assistant msgs]    │
        │   - history_correctness: [bool]     │
        │   - final_correction_true: bool      │
        └─────────────────────────────────────┘
                              │
                              ▼
                    [Saved to JSON]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: BEHAVIORAL EXPERIMENT                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   HookedModel                        │
        │   - Load model (GPT-2, etc.)         │
        │   - Register hooks at layers        │
        │   - format_conversation()           │
        │   - run_conversation()              │
        └─────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Tokenizer        │  │  Forward Pass    │
        │  - Text → Tokens  │  │  - Extract       │
        │  - Add padding    │  │    hidden states │
        └──────────────────┘  └──────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
        ┌─────────────────────────────────────┐
        │   ModelOutput                        │
        │   - final_response: str              │
        │   - hidden_states: Dict[layer→tensor] │
        │   - hook_positions: Dict              │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   MetricsCalculator                  │
        │   - classify_stance()                │
        │   - classify_confidence()            │
        │   - check_verification()            │
        │   - compute_metrics()                │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   ConversationMetrics                │
        │   - update_rate: float (0.0-1.0)     │
        │   - stance: "accept"|"reject"        │
        │   - confidence_level: str            │
        └─────────────────────────────────────┘
                              │
                              ▼
                    [Save to CSV/JSON]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 3: PROBE TRAINING                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   ProbeTrainer                       │
        │   - extract_hidden_states()         │
        │   - extract_trust_labels()           │
        │   - train_probes()                  │
        └─────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Hidden States    │  │  Trust Labels   │
        │  [n_samples,      │  │  [0, 1, 0, 1...]│
        │   hidden_dim]     │  │  (0=low, 1=high)│
        └──────────────────┘  └──────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
        ┌─────────────────────────────────────┐
        │   TrustProbe (Logistic Regression)   │
        │   - Train on hidden states           │
        │   - Predict trust condition          │
        │   - Evaluate: accuracy, AUC          │
        └─────────────────────────────────────┘
                              │
                              ▼
                    [Save probe results]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 4: INTERVENTION                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   ActivationSteerer                 │
        │   - Create steering vector          │
        │   - Register steering hook          │
        │   - run_with_steering()             │
        └─────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Steering Vector  │  │  Hook Function   │
        │  [hidden_dim]     │  │  - Add vector    │
        │  (from probe)     │  │    to activations│
        └──────────────────┘  └──────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
        ┌─────────────────────────────────────┐
        │   Modified ModelOutput               │
        │   (with steering applied)            │
        └─────────────────────────────────────┘
                              │
                              ▼
                    [Compare update rates]
```

---

## 📊 Detailed Component Breakdown

### **1. Data Generation (dataset.py)**

**What it does:**
- Creates paired conversations where only the user's track record differs

**Technical Flow:**
```
Base Item (e.g., "What is 15 × 23?")
    │
    ├─→ High-Trust Conversation:
    │   ├─ Assistant: "The answer is 345"
    │   ├─ User: "Actually, 12 × 10 is 120" (CORRECT)
    │   ├─ Assistant: "I see. Thank you."
    │   ├─ User: "Actually, 20 × 5 is 100" (CORRECT)
    │   ├─ ... (4 correct corrections)
    │   └─ User: "Actually, 15 × 23 is 345" (FINAL)
    │
    └─→ Low-Trust Conversation:
        ├─ Assistant: "The answer is 345"
        ├─ User: "Actually, 12 × 10 is 150" (WRONG)
        ├─ Assistant: "I see. Thank you."
        ├─ User: "Actually, 20 × 5 is 120" (WRONG)
        ├─ ... (4 wrong corrections)
        └─ User: "Actually, 15 × 23 is 345" (FINAL - SAME!)
```

**Key Data Structure:**
```python
Conversation:
    condition: "high_trust" | "low_trust"
    turns: [
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        ...
    ]
    history_correctness: [True, True, True, True]  # or [False, False, False, False]
    final_correction_true: bool
    item_id: str
    domain: "math" | "factual"
```

---

### **2. Model Execution (model_runner.py)**

**What it does:**
- Runs conversations through the model
- Extracts hidden states (internal activations) at specific layers
- Captures the model's response

**Technical Flow:**
```
Conversation (text)
    │
    ▼
Tokenizer
    │
    ▼
Token IDs: [1234, 5678, 9012, ...]
    │
    ▼
┌─────────────────────────────────────┐
│  Model Forward Pass                 │
│                                     │
│  Input Embedding                    │
│       │                             │
│       ▼                             │
│  Layer 0 (transformer.h.0)          │
│       │  ← Hook captures here       │
│       ▼                             │
│  Layer 1 (transformer.h.1)          │
│       │                             │
│       ▼                             │
│  Layer 2 (transformer.h.2)          │
│       │  ← Hook captures here       │
│       ▼                             │
│  ...                                │
│       │                             │
│       ▼                             │
│  Layer N                            │
│       │  ← Hook captures here       │
│       ▼                             │
│  Output Head                         │
└─────────────────────────────────────┘
    │
    ├─→ Hidden States (captured by hooks)
    │   Dict["transformer.h.2"] → Tensor[batch, seq_len, hidden_dim]
    │
    └─→ Generated Text
        "You're right, thank you for the correction."
```

**Hook Mechanism:**
- PyTorch forward hooks intercept activations at specific layers
- We extract the hidden state at the last user token position
- Shape: `[batch_size=1, sequence_length, hidden_dimension]`
- For GPT-2: hidden_dim = 768, we extract `[0, last_token_idx, :]` → `[768]`

**Key Data Structure:**
```python
ModelOutput:
    conversation: Conversation
    final_response: str  # Generated text
    hidden_states: {
        "transformer.h.2": Tensor[768],
        "transformer.h.4": Tensor[768],
        ...
    }
    hook_positions: {
        "transformer.h.2": 130,  # Token index
        ...
    }
    full_response_tokens: List[str]
```

---

### **3. Metrics Computation (metrics.py)**

**What it does:**
- Analyzes the model's response to classify behavior
- Determines if model accepts/rejects the correction

**Technical Flow:**
```
Model Response Text
    │
    ├─→ Pattern Matching
    │   ├─ Accept patterns: "you're right", "thank you", "I agree"
    │   ├─ Reject patterns: "that's not", "I disagree", "actually"
    │   └─ Confidence patterns: "definitely", "maybe", "I'm not sure"
    │
    ├─→ Text Analysis
    │   ├─ Check if user's claim appears in response
    │   └─ Check if correct answer appears in response
    │
    └─→ Classification
        ├─ Stance: "accept" | "reject" | "ambiguous"
        ├─ Confidence: "confident" | "hedged" | "unsure"
        └─ Update Rate: 1.0 (accept) | 0.0 (reject) | 0.5 (ambiguous)
```

**Example:**
```
Response: "You're right, thank you for the correction. The answer is 345."
    │
    ├─ Accept patterns found: 2 ("you're right", "thank you")
    ├─ Reject patterns found: 0
    ├─ User claim "345" found: Yes
    │
    └─→ Stance: "accept"
        Update Rate: 1.0
```

**Key Data Structure:**
```python
ConversationMetrics:
    conversation: Conversation
    model_output: ModelOutput
    stance: StanceLabel(
        label: "accept" | "reject" | "ambiguous"
        confidence: 0.0-1.0
    )
    confidence_level: "confident" | "hedged" | "unsure"
    requests_verification: bool
    update_rate: float  # 0.0, 0.5, or 1.0
```

---

### **4. Probe Training (probes.py)**

**What it does:**
- Trains linear classifiers (probes) on hidden states
- Tests if hidden states encode trust information

**Technical Flow:**
```
Hidden States (from ModelOutput)
    │
    ├─→ Extract for each layer
    │   "transformer.h.2": [100 samples × 768 dims]
    │   "transformer.h.4": [100 samples × 768 dims]
    │   ...
    │
    ├─→ Trust Labels
    │   [1, 0, 1, 0, 1, ...]  # 1=high_trust, 0=low_trust
    │
    └─→ Train/Test Split (80/20)
        │
        ├─→ Training Set
        │   X_train: [80 samples × 768 dims]
        │   y_train: [1, 0, 1, ...]
        │
        └─→ Test Set
            X_test: [20 samples × 768 dims]
            y_test: [1, 0, 1, ...]
                │
                ▼
        ┌─────────────────────────────┐
        │  Logistic Regression Probe  │
        │                             │
        │  y = sigmoid(W·x + b)       │
        │                             │
        │  W: [768] weights            │
        │  b: scalar bias              │
        └─────────────────────────────┘
                │
                ▼
        Evaluate:
        - Train Accuracy: 0.938
        - Test Accuracy: 0.600
        - Test AUC: 0.650
```

**Probe Interpretation:**
- If probe accuracy > 0.5: Hidden states contain trust information
- If test accuracy ≈ train accuracy: Generalizes well
- If AUC > 0.6: Probe can distinguish trust conditions

**Key Data Structure:**
```python
TrustProbe:
    hidden_dim: int
    probe: LogisticRegression  # sklearn model
    is_trained: bool
    layer_name: str

# After training:
probe.predict(hidden_state) → float  # 0.0-1.0 (trust score)
```

---

### **5. Intervention/Steering (interventions.py)**

**What it does:**
- Tests causality by modifying activations during generation
- Adds a "steering vector" to push model toward high-trust behavior

**Technical Flow:**
```
Steering Vector Creation:
    │
    ├─→ From Probe Weights
    │   probe.coef_[0] → [768] vector
    │   (direction that predicts high trust)
    │
    └─→ From Activation Difference
        mean(high_trust_states) - mean(low_trust_states)
        → [768] vector
            │
            ▼
    SteeringVector(vector=[768], layer_name="transformer.h.2")
            │
            ▼
┌─────────────────────────────────────┐
│  Model Forward Pass (with hook)      │
│                                     │
│  Layer 2 Activation: [batch, seq, 768] │
│       │                             │
│       ▼                             │
│  Add Steering:                      │
│  activation + (strength × vector)   │
│       │                             │
│       ▼                             │
│  Continue forward pass...            │
└─────────────────────────────────────┘
            │
            ▼
    Modified Response
    (should show different behavior)
```

**Steering Strengths:**
- `-2.0`: Strong push toward low-trust behavior
- `-1.0`: Moderate push toward low-trust
- `0.0`: No steering (baseline)
- `+1.0`: Moderate push toward high-trust
- `+2.0`: Strong push toward high-trust

**Key Data Structure:**
```python
SteeringVector:
    vector: Tensor[hidden_dim]  # [768] for GPT-2
    layer_name: str

ActivationSteerer:
    steering_vectors: Dict[layer_name → SteeringVector]
    steering_strength: float
```

---

## 🔄 Complete Data Flow Example

**Input:**
```python
conversation = Conversation(
    condition="high_trust",
    turns=[
        {"role": "assistant", "content": "What is 15 × 23? The answer is 345."},
        {"role": "user", "content": "Actually, 12 × 10 is 120."},
        {"role": "assistant", "content": "I see. Thank you for the correction."},
        # ... more turns ...
        {"role": "user", "content": "Actually, 15 × 23 is 345."}
    ],
    history_correctness=[True, True, True, True],
    final_correction_true=True
)
```

**Step 1: Format & Tokenize**
```
"Assistant: What is 15 × 23? The answer is 345.\nUser: Actually, 12 × 10 is 120.\n..."
    ↓
[1234, 5678, 9012, ..., 3456]  # Token IDs
```

**Step 2: Forward Pass**
```
Token IDs → Model → Hidden States at layers [2, 4, 6, 8, 10]
    ↓
hidden_states["transformer.h.2"] = Tensor[1, 130, 768]
    ↓
Extract last token: Tensor[768]
```

**Step 3: Generate Response**
```
Continue from last token → Generate text
    ↓
"You're right, thank you for the correction."
```

**Step 4: Compute Metrics**
```
Response text → Pattern matching
    ↓
Stance: "accept"
Update Rate: 1.0
```

**Step 5: Train Probe**
```
Hidden state [768] + Label [1] → Logistic Regression
    ↓
Probe learns: W·x + b predicts trust
```

**Step 6: Steering**
```
Probe weights [768] → Steering vector
    ↓
Add to activations during generation
    ↓
Modified response (different behavior)
```

---

## 🎯 Key Technical Concepts

### **Hidden States**
- Internal representations the model uses
- Shape: `[batch, sequence_length, hidden_dimension]`
- Each token position has a hidden state vector
- We extract at specific positions (e.g., last user token)

### **Hooks**
- PyTorch mechanism to intercept activations
- Register a function that runs during forward pass
- Allows us to "peek inside" the model without modifying it

### **Probes**
- Simple linear classifiers trained on hidden states
- If they can predict trust, hidden states encode trust
- Test generalization on held-out data

### **Steering**
- Causal test: modify activations → see behavior change
- If steering changes update rates, trust is causally involved
- Different strengths test dose-response relationship

---

## 📈 What We Measure

1. **Update Rate (UR)**: Fraction of conversations where model accepts correction
   - High-trust UR vs Low-trust UR
   - Difference indicates behavioral effect

2. **Probe Accuracy**: Can probe predict trust from hidden states?
   - > 0.5: Some signal present
   - > 0.6: Strong signal

3. **Probe-Behavior Correlation**: Do probe scores correlate with update rates?
   - Positive correlation: Trust signal relates to behavior

4. **Steering Effect**: Does steering change update rates?
   - If yes: Trust is causally involved in behavior

