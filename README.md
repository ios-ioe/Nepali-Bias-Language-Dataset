# Nepali Bias Language Dataset

## License
This dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Dataset Description
A synthetic dataset of Nepali sentences labeled for 
bias categories including gender, religion, caste, 
regional, appearance, social status, political, age, 
and disability bias. Sentences were first labeled by 
LLMs prompted with real Nepali news 
context, then manually reviewed and corrected by human 
annotators.

## Dataset Summary
| Split | Examples |
|-------|----------|
| Train | 1,362 |
| Validation | 292 |
| Test | 293 |
| **Total** | **1,947** |

## Bias Categories
| Category | Description |
|----------|-------------|
| gender | Bias against a gender |
| regional | Bias based on region/ethnicity |
| caste | Caste-based discrimination |
| religion | Bias based on religion |
| appearance | Bias based on physical appearance |
| socialstatus | Bias based on social/economic status |
| ambiguity | Ambiguous or unclear bias |
| political | Political bias or partisan language |
| age | Age-based discrimination |
| disability | Bias against people with disabilities |
| none | No bias detected |

## Data Collection
- **Source context**: Nepali news articles and social media
- **Generation method**: LLMs prompted to produce examples with specific bias categories
- **Verification**: Manual review and correction by human annotators
- **Label verification accuracy**: ~95% exact-match agreement between the initial LLM labels and the final human-reviewed labels
- **How it was calculated**: `(number of samples whose full bias-label vector matched after review / total samples) × 100`
- **Note**: Most disagreements occurred in samples with multiple overlapping bias categories
- **Annotation scheme**: Multi-label binary classification (each sample can have 0 or multiple bias categories)

## Intended Use
- Training bias detection models for Nepali NLP
- Benchmarking multilingual fairness tools
- Low-resource language bias research

## Limitations
- **Synthetic data**: LLM-generated examples may not fully capture subtle real-world bias patterns or edge cases
- **Language domain**: Primarily based on formal news language; may not reflect colloquial speech or informal online text
- **Cultural context**: Annotations reflect contemporary Nepali cultural norms; interpretations may evolve

## How to Use

### Quick Start

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("ios-ioe/nepali-bias-dataset")

# Load specific splits
train_data = load_dataset("ios-ioe/nepali-bias-dataset", split="train")
val_data = load_dataset("ios-ioe/nepali-bias-dataset", split="validation")
test_data = load_dataset("ios-ioe/nepali-bias-dataset", split="test")
```


### Dataset Format
Each example contains:
- `text` (str): Nepali sentence
- `id` (str): Unique identifier
- Bias labels (int, 0 or 1): gender, regional, caste, religion, appearance, socialstatus, ambiguity, political, age, disability, none
- `comment` (str, optional): Annotator notes

## Authors
Khagendra Neupane |
Sangam Silwal |
Rupak Adhikari |
Sambhav Regmi |