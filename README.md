# 3-Channel LLM Watermarking for Hangul Jamo Structure

This is a **capstone project** to suggest a **Korean specific LLM watermarking strategy** by @mirulili at **Yonsei University (2025 Fall)**.

<br>

## Structure

```
3Ch-Jamo-Watermark/
├─ src/
│  ├─ __init__.py
│  ├─ main.py                           # Execute watermark generation and detection pipeline
│  │
│  ├─ model/                            # Language model related modules
│  │  ├─ __init__.py
│  │  ├─ load_model.py                   # Load model and tokenizer
│  │  └─ generate.py                     # Text generation logic
│  │
│  ├─ watermark/                        # Core watermarking logic modules
│  │  ├─ __init__.py
│  │  ├─ jamo_utils.py                   # Hangul Jamo decomposition utility
│  │  ├─ payload_mgr.py                  # Manage message <-> bit sequence conversion
│  │  ├─ hash_policy.py                  # Jamo-based hash calculation policy
│  │  ├─ processor.py                    # JamoWatermarkProcessor (Watermark insertion)
│  │  └─ detector.py                     # JamoWatermarkDetector (Watermark detection)
│  │
│  └─ evaluation/                       # Performance evaluation related modules
│     ├─ __init__.py
│     ├─ eval_quality.py                 # Measure generation quality (PPL, etc.)
│     └─ eval_robustness.py              # Robustness testing
│
├─ .gitignore                          
├─ environment.yml
├─ Makefile                            
└─ README.md
```

<br>


## Execution Instructions

1.  **Create and activate a conda environment**:
    ```bash
    make setup
    conda activate jamo
    ```
2.  **Install Dependencies**:
    ```bash
    make install
    ```
    The environment is defined in `environment.yml`. Use `make setup` for first-time creation and `make install` to update an existing environment.
3.  **Run Program**:
    ```bash
    make run
    ```
    This executes `src/main.py`, which performs the **entire process** of inserting the watermark to generate text and then restoring the message from the generated text.
4. **Test Robustness**:
    ```bash
    make test_robustness
    ```

<br>

## Core Operating Principle

1.  **Jamo Channel Separation**: Decompose a Hangul syllable into **three channels** -- Choseong (initial consonant), Jungseong (medial vowel), and Jongseong (final consonant) -- and **independently assign** a watermark bit to each channel.
2.  **Round-Robin Channel Selection & Target Bit Matching**: At each watermarking insertion step, **one channel is selected in a deterministic round-robin order** (`channel_idx = step_t % 3` in robustness mode; `1 + step_t % 2` in quality mode, which skips Choseong to put more weight on vowels and final consonants). Generator and detector follow the same rule, so they stay synchronized without sharing a key. Calculate a hash value from the Jamo indices of each token and check if this value matches the **target bit** for the current step.
3.  **Conditional Step Synchronization**:
      * **Insertion (Processor)**: After applying a bias to the logits (probabilities), the watermark is considered inserted and moves to the next bit (**`step_t` increments**) **only if the most likely candidate token** matches the target bit and is actually selected.
      * **Detection (Detector)**: Read the tokens of the generated text sequentially, and extract the watermark and move to the next bit (**`step_t` increments**) **only if the token's hash value** matches the **target bit** to be found.
      * This method allows the generator and detector to maintain **synchronization** by advancing the step according to the same rule, despite the uncertainty of sampling.

<br>

## Preview of Detection Results

### Watermarked Text
<img width="440" alt="watermarked" src="https://github.com/user-attachments/assets/ae806f60-a313-43ae-9075-6488491825be" />

### General Text
<img width="450" alt="unwatermarked" src="https://github.com/user-attachments/assets/963f6e31-602b-4db3-a04f-79e3e3c26189" />

(Used [MarkLLM Toolkit](https://github.com/THU-BPM/MarkLLM) visualizer)

<br>

Full report can be found [here](https://github.com/mirulili/3Ch-Jamo-Watermark/blob/main/Report.pdf).
