# Plagiarism Detector

A robust, command-line tool for detecting plagiarism in text documents using locality-sensitive hashing (LSH) and edit distance algorithms.

## Overview

This plagiarism detection tool analyzes text documents to identify potential instances of plagiarism by measuring content similarity. The system uses a combination of Locality-Sensitive Hashing (LSH) and Levenshtein edit distance to calculate similarity scores between documents or individual sentences.

## Features

- **Multiple Detection Methods**: Combines LSH and edit distance for improved accuracy
- **Three Operating Modes**:
  - Classic mode: Compares sentences from an input file
  - File comparison: Direct comparison between two specific files
  - Directory mode: Compares all file pairs in a directory
- **Flexible Thresholds**: Configurable similarity thresholds for fine-tuning detection
- **Detailed Reports**: Generates comprehensive plagiarism reports with similarity scores
- **Case-Insensitive**: Converts text to lowercase for more accurate comparisons
- **Sentence-Level Analysis**: Splits documents into sentences for granular comparison

## Technical Implementation

The system is implemented in C and uses the following key techniques:

- **K-Shingles**: Breaks text into overlapping k-character sequences
- **MinHash**: Efficiently approximates Jaccard similarity with dense vectors
- **LSH**: Locality-Sensitive Hashing for scalable similarity detection
- **Levenshtein Distance**: Measures edit operations needed to transform one string to another
- **Cosine Similarity**: Calculates vector similarity for comparison

## Installation

### Prerequisites

- GCC compiler or compatible C compiler
- Standard C libraries

### Compilation

```bash
gcc -o plagiarism_detector plagiarism_detector.c -lm
```
OR
```bash
# Building: Use the provided `Makefile` to build the project. In the terminal, run:
make

# Cleaning : To remove compiled files, use:
make clean
```

Note: The `-lm` flag is required for linking the math library.

## Usage

### Classic Mode

```bash
./plagiarism_detector
```

This mode reads input from `Input.txt` and writes results to `Output.txt`.

### File Comparison Mode

```bash
./plagiarism_detector -f <file1> <file2>
```

This compares two specific files and writes a report to `PlagiarismReport.txt`.

### Directory Mode

```bash
./plagiarism_detector -d <directory_path>
```

This compares all pairs of files in the specified directory and writes a report to `PlagiarismReport.txt`.

## Input and Output

### Input Formats

- **Classic Mode**: Expects an `Input.txt` file with numbered sentences (e.g., "1. This is a sentence.")
- **File/Directory Modes**: Accepts text documents in any format

### Output Formats

- **Classic Mode**: Creates `Output.txt` with edit distances and similarity scores between sentences
- **File/Directory Modes**: Generates `PlagiarismReport.txt` with detailed plagiarism analysis

## Command-Line Options

- `-f <file1> <file2>`: Compare two specific files
- `-d <directory_path>`: Compare all files in a directory
- No options: Run in classic mode

## Algorithm Details

### Document Processing Flow

1. **Text Preprocessing**:
   - Convert to lowercase
   - Split into sentences based on punctuation (., !, ?)
   - Remove leading/trailing whitespace

2. **Shingle Creation**:
   - Generate k-shingles (default k=3)
   - Create sets of unique shingles per sentence
   - Build vocabulary from all unique shingles

3. **Vector Representation**:
   - Create one-hot encoding vectors based on vocabulary
   - Generate dense MinHash vectors for efficient comparison

4. **Similarity Calculation**:
   - LSH similarity: Cosine similarity between dense vectors
   - Edit distance similarity: Normalized Levenshtein distance
   - Overall similarity: Average of both methods

5. **Plagiarism Detection**:
   - Compare similarity scores against thresholds
   - Generate detailed reports of findings

### Key Parameters

- `SIMILARITY_THRESHOLD` (default: 0.7): Threshold for sentence similarity
- `PLAGIARISM_THRESHOLD` (default: 0.5): Threshold for document-level plagiarism
- `MAX_SENTENCES` (default: 1000): Maximum number of sentences to process
- `MAX_LINE_LENGTH` (default: 1024): Maximum length of a single line

These parameters can be adjusted in the code to fine-tune detection sensitivity.

## Performance Considerations

- The LSH algorithm significantly improves performance for large documents compared to naive pairwise comparison
- Processing time increases quadratically with the number of sentences
- Memory usage depends on document size and vocabulary size
- For very large documents, consider increasing `MAX_SENTENCES` and memory allocation sizes

## Limitations

- Limited support for non-Latin character sets
- No support for detecting paraphrasing or idea plagiarism
- Memory allocation is static and may need adjustment for very large documents
- Simple sentence splitting based on punctuation (may not handle all cases correctly)
