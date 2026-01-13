# TeamRules: A prototype ReV-AI advising method

This repository contains the implementation of the **TeamRules algorithm**. 

## Quick Start

### Main Algorithm
The **TeamRules algorithm** implementation is found in `tr.py`. This contains the core logic for:
- Generating interpretable rule-based advice
- Modeling human acceptance/rejection of AI recommendations  
- Optimizing when to provide advice vs. withhold it

### Running Experiments
To reproduce the experimental results, run the provided shell scripts:

```bash
./run_cost0.sh    # For contradiction cost = 0.0
./run_cost01.sh   # For contradiction cost = 0.1
# etc.
```

### Command Line Arguments
Each experiment line follows this format:
```bash
nohup python3 -u main.py -d 'fico' -i 10 -h 'biased' -r 'standard' -c 0.0 -n False -p '' -b False -w 'brs,tr-no(ADB),tr' > ../logs/log10_fico_biased_standard_cost00.out &
```

**Parameters:**
- `-d 'fico'`: Dataset (fico, heart_disease, hr, adult)
- `-i 10`: Run number (each run uses different random seed)
- `-h 'biased'`: Human type (calibrated, biased, offset_01, etc.)
- `-r 'standard'`: Run type configuration  
- `-c 0.0`: Contradiction cost/penalty weight
- `-n False`: Whether to remake human models (False = use cached)
- `-p ''`: Custom name suffix for output files
- `-b False`: Whether human has decision bias
- `-w 'brs,tr-no(ADB),tr'`: Which models to train (BRS, TeamRules without ADB, full TeamRules)

### Analyzing Results
After training models, use `analysis_create.py` to generate performance metrics and comparisons:

```bash
python analysis_create.py
```

This script:
- Loads trained models from the `results/` directory
- Evaluates them on test data
- Generates comparative analysis across different methods and settings

### Output
Results are saved to:
- `results/[dataset]/run[i]/` - Trained models and human simulations
- `logs/` - Execution logs

### Human Behavior Settings (Paper Mapping)
The code supports three main human behavior settings described in the paper:

Difficulty-biased decisions + Group-biased ADB:
-h 'biased' + -b False + -p ''
Group-biased decisions + Group-biased ADB:
-h 'biased' + -b True + -p '_dec_bias'
Difficulty-biased decisions + Accuracy-biased ADB:
-h 'offset_01' + -b False + -p ''

### Models Compared
- **BRS**: Baseline Task-Only Method (Bayesian Rule Sets, accuracy-focused)
- **TR**: TeamRules (our method - human-aware with advice filtering)
- **TR-no(ADB)**: TeamRules without human acceptance modeling

TR-no(ADB, Cost) and TR-no(Cost) results are produced by evaluating the TR-no(ADB) and TR methods trained for cost 0 (assuming no cost) across all cost settings. 