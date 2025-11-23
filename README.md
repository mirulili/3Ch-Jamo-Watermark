# 3-Channel LLM Watermarking for Hangul Jamo Structure

This is a **capstone project** to suggest a **Korean specific LLM watermarking strategy** at **Yonsei University (2025 Fall)**.



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
│     ├─ eval_quality.py                 # (Planned) Measure generation quality (PPL, etc.)
│     └─ eval_robustness.py              # (Currently disabled) Robustness testing
│
├─ .gitignore                          # Git tracking exclusion settings
├─ Makefile                            # Automation for build and execution
├─ README.md                           # Project description
└─ requirements.txt                    # List of dependency libraries
```




## Execution Instructions

1.  **Install Dependencies**:
    ```bash
    make install
    ```
2.  **Run Program**:
    ```bash
    make run
    ```
    This executes `src/main.py`, which performs the **entire process** of inserting the watermark to generate text and then restoring the message from the generated text.
3. **Test Robustness**:
    ```bash
    make test_robustness
    ```


## Core Operating Principle

1.  **Jamo Channel Separation**: Decompose a Hangul syllable into **three channels** -- Choseong (initial consonant), Jungseong (medial vowel), and Jongseong (final consonant) -- and **independently assign** a watermark bit to each channel.
2.  **Parallel Channel Selection & Target Bit Matching**: At each watermarking insertion step, **one channel among three is randomly selected**. Calculate a hash value from the Jamo indices of each token and check if this value matches the **target bit** for the current step.
3.  **Conditional Step Synchronization**:
      * **Insertion (Processor)**: After applying a bias to the logits (probabilities), the watermark is considered inserted and moves to the next bit (**`step_t` increments**) **only if the most likely candidate token** matches the target bit.
      * **Detection (Detector)**: Read the tokens of the generated text sequentially, and extract the watermark and move to the next bit (**`step_t` increments**) **only if the token's hash value** matches the **target bit** to be found.
      * This method allows the generator and detector to maintain **synchronization** by advancing the step according to the same rule, despite the uncertainty of sampling.
