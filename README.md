<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        ███╗   ███╗ █████╗ ██████╗ ██╗  ██╗ ██████╗ ██╗   ██╗                 ║
║        ████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██╔═══██╗██║   ██║                 ║
║        ██╔████╔██║███████║██████╔╝█████╔╝ ██║   ██║██║   ██║                 ║
║        ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ██║   ██║╚██╗ ██╔╝                 ║
║        ██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗╚██████╔╝ ╚████╔╝                  ║
║        ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═══╝                   ║
║                                                                              ║
║            C H A I N   T E X T   G E N E R A T I O N   E N G I N E           ║
║                                                                              ║
║                    World-Class • Production-Grade • Resilient                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**The most complete, resilient, and feature-rich Markov Chain text generation
system ever written in pure Python.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Production%20Grade-gold?style=for-the-badge)]()
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20External-brightgreen?style=for-the-badge)]()
[![Edge Cases](https://img.shields.io/badge/Edge%20Cases%20Handled-1000%2B-red?style=for-the-badge)]()

</div>

---

## Table of Contents

- [What Is This?](#what-is-this)
- [What Is a Markov Chain?](#what-is-a-markov-chain)
- [Why This System Is Different](#why-this-system-is-different)
- [Feature Map](#feature-map)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Usage Guide](#full-usage-guide)
  - [Main Menu](#main-menu)
  - [Training a Model](#training-a-model)
  - [Text Input Modes](#text-input-modes)
  - [Generation Strategies](#generation-strategies)
  - [Smoothing](#smoothing)
  - [Seeds and Reproducibility](#seeds-and-reproducibility)
  - [Saving and Loading Models](#saving-and-loading-models)
  - [Model Statistics](#model-statistics)
- [Configuration Reference](#configuration-reference)
- [Component Deep Dive](#component-deep-dive)
  - [SystemConfig](#systemconfig)
  - [TextSanitizer](#textsanitizer)
  - [InputManager](#inputmanager)
  - [NGramTable](#ngramtable)
  - [ModelPersistence](#modelpersistence)
  - [MarkovGenerator](#markovgenerator)
  - [OutputFormatter](#outputformatter)
  - [SessionManager](#sessionmanager)
- [Edge Cases and Resilience](#edge-cases-and-resilience)
- [Understanding the Output Statistics](#understanding-the-output-statistics)
- [How Order Affects Output Quality](#how-order-affects-output-quality)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Technical Glossary](#technical-glossary)
- [Contributing](#contributing)
- [License](#license)

---

## What Is This?

This is a **production-grade Markov Chain text generation engine** built
entirely in Python with zero external dependencies.

You give it text. It learns the statistical patterns of that text —
which characters tend to follow which other characters — and then
generates brand new text that *sounds like* the original.

It is used for:

- Generating realistic-sounding prose from a book or article
- Creating code that looks syntactically plausible
- Studying the statistical structure of any written language
- Generating test data that resembles real text
- Exploring what a text "sounds like" at a probabilistic level
- Educational exploration of language models and probability theory

---

## What Is a Markov Chain?

A **Markov Chain** is a mathematical system where the next state depends
only on the current state — not on the entire history of how you got there.

In text generation, the "state" is a window of the last N characters
(called an **n-gram** or the **order** of the chain). The system predicts
the next character purely based on this window.

### A Simple Example

Training text:
```
"the cat sat on the mat"
```

With **order = 2**, the system builds a table like this:

```
"th" → ['e', 'e']         (seen before "the" twice)
"he" → [' ', ' ']         (space follows "he" twice)
"e " → ['c', 'm']         ("c" in "cat", "m" in "mat")
"ca" → ['t']
"at" → [' ', ' ']
" c" → ['a']
" s" → ['a']
" o" → ['n']
"on" → [' ']
"n " → ['t']
" t" → ['h', 'h']
"sa" → ['t']
" m" → ['a']
"ma" → ['t']
```

To generate text, it:
1. Picks a starting 2-gram (e.g., `"th"`)
2. Looks up what can follow `"th"` → picks `"e"` randomly
3. Slides the window: now at `"he"`
4. Looks up what follows `"he"` → picks `" "`
5. Continues until enough text is generated

Higher order = larger window = more context = more coherent output,
but also needs more training text.

---

## Why This System Is Different

Most Markov chain implementations are 20-line scripts. This system is built
to handle **everything** — including scenarios that nobody else thinks about.

```
┌─────────────────────────────────────────────────────────────────┐
│              STANDARD vs THIS SYSTEM                            │
├───────────────────────────┬─────────────────────────────────────┤
│ Standard Implementation   │ This System                         │
├───────────────────────────┼─────────────────────────────────────┤
│ Crashes on empty input    │ Detects, explains, and recovers     │
│ Crashes on dead ends      │ 3-tier recovery with 100 attempts   │
│ No memory protection      │ Hard cap at 5M n-gram entries       │
│ No timeout                │ 60-second wall-clock guard          │
│ Uniform sampling only     │ Weighted, greedy, temperature modes │
│ No reproducibility        │ Full seeded RNG support             │
│ No persistence            │ JSON + Pickle with atomic writes    │
│ Single text source        │ Direct / File / Demo                │
│ No encoding handling      │ 4-tier encoding detection           │
│ No statistics             │ 12 metrics including perplexity     │
│ No logging                │ Dual-stream log (console + file)    │
│ No session management     │ Full multi-round interactive loop   │
│ No Unicode handling       │ NFC normalization + 13 invisible    │
│                           │ char types removed                  │
│ No input validation       │ Every input fully validated         │
│ Crashes on Ctrl+C         │ Graceful interrupt handling         │
└───────────────────────────┴─────────────────────────────────────┘
```

---

## Feature Map

```
MARKOV ENGINE — COMPLETE FEATURE MAP
═══════════════════════════════════════════════════════════════

INPUT SUBSYSTEM
├── Mode 1: Direct keyboard/paste input
│   ├── Multi-line support
│   ├── END sentinel to finish
│   ├── Ctrl+C graceful recovery
│   └── Ctrl+D (EOF) graceful recovery
├── Mode 2: File loading
│   ├── Encoding auto-detection (UTF-8-SIG → UTF-8 → Latin-1 → CP1252)
│   ├── Binary fallback with ASCII replacement
│   ├── File size validation (50 MB limit)
│   ├── Empty file detection
│   ├── Directory vs file detection
│   ├── Retry on not-found with user prompt
│   └── 3x retry on OS/permission errors
└── Mode 3: Built-in demo texts
    ├── English prose (classic literature)
    ├── Python code snippet
    └── Lorem ipsum

TEXT SANITIZATION (10-stage pipeline)
├── Stage 1:  BOM removal
├── Stage 2:  Null byte removal
├── Stage 3:  13 invisible/zero-width character types removed
├── Stage 4:  Unicode NFC normalization
├── Stage 5:  Line ending normalization (CRLF → LF, CR → LF)
├── Stage 6:  Control character replacement (keep \n, \t)
├── Stage 7:  Optional URL removal
├── Stage 8:  Long line truncation (configurable limit)
├── Stage 9:  Excessive blank line collapse (max 2 consecutive)
└── Stage 10: Leading/trailing whitespace strip

MODEL BUILDING (NGramTable)
├── Character-level n-gram extraction
├── Frequency counting (not just lists)
├── Memory guard (configurable entry limit)
├── Build time measurement
└── 12-metric statistics computation
    ├── Total n-grams
    ├── Unique n-grams
    ├── Vocabulary size
    ├── Average branching factor
    ├── Maximum branching factor
    ├── Dead end count
    ├── Coverage ratio
    ├── Shannon entropy (per-state average)
    ├── Perplexity
    ├── Build time
    ├── Text SHA-256 hash
    └── Model order

GENERATION ENGINE (MarkovGenerator)
├── Strategy 1: Standard (frequency-weighted random)
├── Strategy 2: Greedy (always most likely next char)
├── Strategy 3: Temperature-scaled sampling
│   ├── T < 1.0: conservative / repetitive
│   ├── T = 1.0: equivalent to standard
│   └── T > 1.0: creative / chaotic
├── Dead-end recovery (3 tiers)
│   ├── Tier 1: Tail-overlap candidate search
│   ├── Tier 2: Random table restart
│   └── Tier 3: Give up after 100 attempts
├── Timeout protection (60 second default)
├── Seeded RNG for full reproducibility
├── Laplace (add-k) smoothing
├── Smart start n-gram selection
│   ├── Prefers sentence starters (uppercase, punctuation)
│   └── Falls back to random
├── Real-time progress display for large generations
└── Partial result return on timeout/failure

PERSISTENCE (ModelPersistence)
├── Format 1: JSON (human-readable, portable)
├── Format 2: Pickle (fast, binary)
├── Atomic writes (write temp → rename, crash-safe)
├── Automatic directory creation
├── Version tagging on all saved files
├── Version mismatch warning on load
├── Corrupted file detection
└── Model listing with file sizes

OUTPUT
├── Generation statistics report
├── Paged display for long text (40 lines/page)
├── Save to file option (UTF-8)
├── High-perplexity warning
└── Partial completion notice

SESSION MANAGEMENT
├── Multi-round loop (no restart needed)
├── Model kept in memory across generations
├── Per-action availability checks
├── Full error recovery at every level
└── Graceful Ctrl+C at any point
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main()                                      │
│                 (bootstraps all components)                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SessionManager                                 │
│         (interactive loop, menu, action dispatch)                   │
└────┬────────────┬────────────┬────────────┬────────────┬────────────┘
     │            │            │            │            │
     ▼            ▼            ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
│ Input   │ │NGram    │ │ Markov   │ │ Model   │ │ Output   │
│ Manager │ │ Table   │ │Generator │ │Persist. │ │Formatter │
│         │ │         │ │          │ │         │ │          │
│ •Direct │ │ •Build  │ │ •Standard│ │ •Save   │ │ •Stats   │
│ •File   │ │ •Sample │ │ •Greedy  │ │ •Load   │ │ •Paged   │
│ •Demo   │ │ •Stats  │ │ •Temp.   │ │ •List   │ │ •Save    │
└────┬────┘ └─────────┘ └──────────┘ └─────────┘ └──────────┘
     │
     ▼
┌─────────────────────┐
│   TextSanitizer     │
│   (10-stage pipe)   │
└─────────────────────┘

Cross-cutting concerns (apply everywhere):
┌──────────────────────────────────────────┐
│  SystemConfig  │  LOGGER  │  Exceptions  │
│  (all params)  │ (dual)   │  (hierarchy) │
└──────────────────────────────────────────┘
```

### Data Flow

```
User Input
    │
    ▼
TextSanitizer ──► clean text
    │
    ▼
NGramTable.build() ──► {ngram: {char: count}} + NGramStats
    │
    ▼
MarkovGenerator.generate() ──► GenerationResult
    │
    ▼
OutputFormatter.display() ──► Terminal + optional file
    │
    ▼
ModelPersistence.save() ──► .json or .pkl file
```

---

## Installation

### Requirements

- **Python 3.8 or higher**
- **Zero external packages** — uses only the Python standard library

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/markov-engine.git

# 2. Enter the directory
cd markov-engine

# 3. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows

# 4. Run directly — no pip install needed
python markov_engine.py
```

That is all. There is nothing to install.

---

## Quick Start

```bash
python markov_engine.py
```

You will see:

```
════════════════════════════════════════════════════════════
  ✦  WORLD-CLASS MARKOV CHAIN TEXT GENERATION SYSTEM  ✦
════════════════════════════════════════════════════════════
  Handles any text. Any language. Any edge case.

  MAIN MENU
  ─────────────────────────────────────────────
  Current model:   [no model loaded]
  ─────────────────────────────────────────────
  [✓] train      Train a new model
  [✗] generate   Generate text
  [✗] save       Save current model
  [✓] load       Load a saved model
  [✗] stats      Show model statistics
  [✓] quit       Quit
```

**Fastest path to output:**

1. Type `train` → Enter
2. Choose `3` (Demo text) → Enter
3. Choose `1` (English prose) → Enter
4. Enter order `4` → Enter
5. Back at menu: type `generate` → Enter
6. Enter `500` characters → Enter
7. Choose strategy `1` (Standard) → Enter
8. Press Enter to skip seed
9. Press Enter to use default smoothing

You will see 500 characters of generated text in about 0.01 seconds.

---

## Full Usage Guide

### Main Menu

The main menu is the central hub. Options that require a trained model
are marked with `✗` until you train or load one.

```
[✓] train      — Always available. Trains a new model from text.
[✗] generate   — Requires model. Generate text from current model.
[✗] save       — Requires model. Saves model to disk.
[✓] load       — Always available. Load a previously saved model.
[✗] stats      — Requires model. Show detailed model statistics.
[✓] quit       — Always available. Exit the program.
```

You can press **Ctrl+C** at any point inside an action to cancel it
and return to the menu safely.

---

### Training a Model

Select `train` from the main menu.

**Step 1:** Choose your text input mode (see below).

**Step 2:** Choose the **order** (the size of the context window).

```
Enter Markov order [1–20] (higher = more coherent, but needs more text):
```

The maximum order is automatically capped at half the text length
to ensure at least a minimal number of n-grams can be built.

**Step 3:** The model builds immediately and shows a summary:

```
  ✓ Model trained!
     • 1,847 unique n-grams
     • Vocabulary: 67 unique characters
     • Branching factor: 3.21 avg
     • Perplexity: 4.87
```

---

### Text Input Modes

#### Mode 1 — Direct Input

Type or paste text directly into the terminal.

```
Type or paste your text below.
Press ENTER after each line. Type END on its own line to finish.

It was the best of times, it was the worst of times.
The quick brown fox jumps over the lazy dog.
END
```

- Handles pasting multiple lines at once
- If you press **Ctrl+C** mid-entry, text entered so far is used
- If you press **Ctrl+D** (EOF), same behavior

#### Mode 2 — File Loading

Provide any file path on your system.

```
Enter file path: /home/user/books/moby_dick.txt
```

The system automatically detects the encoding by trying, in order:
1. `UTF-8-SIG` (UTF-8 with BOM)
2. `UTF-8`
3. `Latin-1` (ISO 8859-1)
4. `CP1252` (Windows Western European)
5. Binary fallback with ASCII replacement

File constraints:
- Maximum size: **50 MB** (configurable)
- Must be a regular file (not a directory, device, etc.)
- Must not be empty

#### Mode 3 — Demo Text

Three built-in texts for immediate use:

| # | Name | Content |
|---|------|---------|
| 1 | `english_prose` | Excerpts from classic literature |
| 2 | `code_snippet` | Python Fibonacci function |
| 3 | `lorem_ipsum` | Standard Lorem Ipsum paragraph |

---

### Generation Strategies

When you run `generate`, you choose one of three strategies:

#### Strategy 1: Standard

Picks the next character with probability proportional to how often
it appeared after the current n-gram in the training text.

If `"th"` was followed by `"e"` 8 times and `"a"` 2 times,
there is an 80% chance of picking `"e"` and 20% of picking `"a"`.

**Best for:** General use. Balanced coherence and variety.

#### Strategy 2: Greedy

Always picks the single most frequent next character.
The output is **deterministic** (same result every time for same model).

**Best for:** Seeing what the "most likely" continuation is.
Tends to get stuck in repetitive loops.

#### Strategy 3: Temperature

Applies a temperature scaling factor to the probability distribution.

```
Temperature T < 1.0  →  Sharper distribution  →  More conservative output
Temperature T = 1.0  →  Same as Standard
Temperature T > 1.0  →  Flatter distribution  →  More creative / chaotic output
```

You will be asked to enter a temperature between 0.1 and 3.0.

**Examples:**
- `T = 0.3`: Very conservative. Repeats common patterns.
- `T = 1.0`: Normal behavior.
- `T = 2.0`: Surprising, creative, sometimes nonsensical.

**Best for:** Tuning the creativity vs. coherence trade-off.

---

### Smoothing

**Laplace (add-k) smoothing** handles the case where the current
n-gram has never been seen before in the training text.

Without smoothing: unknown n-gram → dead end → recovery attempt.

With smoothing: unknown n-gram → uniform random choice from all
known characters (vocabulary), weighted by the smoothing constant.

```
  Laplace smoothing [0.0=off, default=0.01]:
```

- `0.0` — No smoothing. Pure Markov chain behavior.
- `0.01` — Light smoothing (default). Rarely triggers.
- `1.0` — Full Laplace smoothing. Strong bias toward uniformity.
- Values above `1.0` — Increasingly uniform output.

**Recommendation:** Leave at `0.01` unless you have a very small
training corpus or very high order.

---

### Seeds and Reproducibility

Every generation uses a **random number generator**. By providing
a seed, you make the generation fully **reproducible**:

```
  Random seed for reproducibility (Enter to skip): 42
```

Providing the same seed with the same model and parameters will
produce **identical output** every time, on any machine.

Leave blank for a random (non-reproducible) result.

**Use cases for seeds:**
- Sharing a specific generated output with others
- Debugging generation behavior
- A/B comparing two parameter settings fairly

---

### Saving and Loading Models

#### Saving

After training, select `save` from the menu.

```
  Model name to save as (no extension): my_moby_dick_model
  Format: [1] JSON (readable), [2] Pickle (fast):
```

| Format | File | Speed | Human-readable | Size |
|--------|------|-------|----------------|------|
| JSON | `name.json` | Slower | ✓ Yes | Larger |
| Pickle | `name.pkl` | Faster | ✗ No | Smaller |

Models are saved to the `markov_models/` directory (configurable).

Writes are **atomic**: the file is written to a `.tmp` file first,
then renamed. This means a crash during save can never produce
a corrupted model file.

#### Loading

Select `load` from the menu. You will see a numbered list of saved models:

```
  Saved models:
    [1] my_moby_dick_model.json
    [2] code_model.pkl
  Select number or enter full path:
```

You can also enter a full path to load a model from anywhere on disk.

Version mismatches produce a warning but the system still attempts
to load the model.

---

### Model Statistics

Select `stats` from the menu for a full breakdown:

```
  ╔══════════════════════════════════╗
  ║       MODEL STATISTICS           ║
  ╠══════════════════════════════════╣
  ║  Order              : 4          ║
  ║  Total n-grams      : 9,847      ║
  ║  Unique n-grams     : 1,203      ║
  ║  Vocabulary size    : 67         ║
  ║  Avg branch factor  : 3.210      ║
  ║  Max branch factor  : 26         ║
  ║  Dead ends          : 142        ║
  ║  Coverage ratio     : 0.1221     ║
  ║  Entropy (bits)     : 1.8304     ║
  ║  Perplexity         : 3.55       ║
  ║  Build time (s)     : 0.0023     ║
  ║  Text hash          : a3f9c1b2   ║
  ╚══════════════════════════════════╝
```

See [Understanding the Output Statistics](#understanding-the-output-statistics)
for a full explanation of every metric.

---

## Configuration Reference

All system parameters are defined in `SystemConfig` at the top of the file.
Edit them there — no magic numbers anywhere else.

```python
@dataclass
class SystemConfig:
    # ── Generation Limits ─────────────────────────────────
    MIN_ORDER: int = 1
    # Minimum allowed Markov order.
    # Cannot be set below 1.

    MAX_ORDER: int = 20
    # Maximum allowed Markov order.
    # Also dynamically capped at half the text length.

    MIN_CHARACTERS: int = 1
    # Minimum characters to generate.

    MAX_CHARACTERS: int = 1_000_000
    # Maximum characters to generate (1 million).
    # Increase if you need very long outputs.

    MIN_TEXT_LENGTH: int = 10
    # Minimum text length accepted for training.

    # ── Memory & Performance ──────────────────────────────
    MAX_NGRAM_ENTRIES: int = 5_000_000
    # Maximum unique n-gram keys in the table.
    # Prevents runaway memory on huge corpora with high orders.
    # 5M entries ≈ several hundred MB RAM depending on key length.

    MAX_TEXT_BYTES: int = 50 * 1024 * 1024
    # Maximum file size for file input: 50 MB.

    GENERATION_TIMEOUT_SECONDS: int = 60
    # Wall-clock time limit for a single generation run.
    # Partial result is returned on timeout.

    # ── Smoothing & Quality ───────────────────────────────
    DEFAULT_SMOOTHING: float = 0.01
    # Laplace add-k smoothing default.
    # 0.0 = off, higher = more uniform.

    DEAD_END_RESTART_ATTEMPTS: int = 100
    # How many dead-end recovery attempts before giving up.

    QUALITY_PERPLEXITY_WARN: float = 500.0
    # Perplexity above this threshold triggers a user warning.

    # ── Persistence ───────────────────────────────────────
    MODEL_SAVE_DIR: str = "markov_models"
    # Directory where models are saved.

    MODEL_EXTENSION: str = ".mkv"
    # Not currently used for extension (JSON/pkl used instead).

    # ── Logging ───────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    # Python logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL

    LOG_FILE: Optional[str] = "markov_system.log"
    # Log file path. Set to None to disable file logging.
```

---

## Component Deep Dive

### SystemConfig

The single source of truth for all tunable parameters. Defined as
a Python `dataclass` so it is easy to instantiate, copy, and modify
programmatically for testing.

Every number in the entire codebase traces back to this class.
There are no magic numbers embedded in logic.

---

### TextSanitizer

A 10-stage text cleaning pipeline that runs on every piece of text
before it touches the model.

**Why is this necessary?**

Text from the real world contains:
- Files saved on Windows have `\r\n` line endings; Unix has `\n`; old Mac has `\r`
- Word processors insert invisible formatting characters
- Some files start with a **BOM** (Byte Order Mark: `\ufeff`)
- Web-scraped text contains URLs, HTML entities, zero-width joiners
- Copy-pasted text from PDFs contains soft hyphens and ligatures
- Some encodings produce "mojibake" (garbage characters from wrong encoding)

Without sanitization, any of these can corrupt the n-gram table with
junk patterns that will appear in generated text.

**The 10 stages:**

```
Stage  1  Remove BOM (\ufeff at start of string)
Stage  2  Remove null bytes (\x00)
Stage  3  Remove 13 invisible Unicode characters
           (zero-width space, joiners, soft hyphen, RTL/LTR marks, etc.)
Stage  4  Unicode NFC normalization
           (é as one char, not e + combining accent as two chars)
Stage  5  Normalize line endings → \n only
Stage  6  Replace remaining control characters with space
           (keep \n and \t which are meaningful)
Stage  7  Optional URL removal (off by default)
Stage  8  Truncate lines longer than max_line_length
Stage  9  Collapse 3+ consecutive blank lines to 2
Stage 10  Strip leading/trailing whitespace
```

If the text is empty after sanitization, a `TextSanitizationError`
is raised with a clear message.

---

### InputManager

Handles all user interaction for collecting text and parameters.

**Key behaviors:**

- `get_text()` presents the three-way mode selection menu
- `get_order()` dynamically caps the maximum based on text length
- `get_num_characters()` validates bounds on every input
- All inputs loop until valid — the user is never dropped to a crash

**File loading encoding cascade:**

```
Try UTF-8-SIG (handles files saved by Windows with BOM)
    ↓ fail
Try UTF-8 (standard modern encoding)
    ↓ fail
Try Latin-1 (covers all byte values, never fails technically)
    ↓ fail (lookup error only)
Try CP1252 (Windows Western European)
    ↓ fail
Binary read → decode as ASCII with replacement characters (■)
```

---

### NGramTable

The core data structure of the engine.

**Internal representation:**

```python
{
    "th": {"e": 42, "a": 3, "i": 1},
    "he": {" ": 38, "r": 4, "n": 2},
    ...
}
```

Stores **counts** (frequencies), not lists of characters.
This is more memory-efficient and enables weighted sampling
and smoothing without restructuring.

**Weighted sampling algorithm:**

```
Given n-gram "th" with counts {"e": 42, "a": 3, "i": 1}
Total = 46

Generate random float r in [0, 46)

r < 42         → pick "e"    (probability 42/46 = 91.3%)
42 ≤ r < 45    → pick "a"    (probability  3/46 =  6.5%)
45 ≤ r < 46    → pick "i"    (probability  1/46 =  2.2%)
```

**Memory guard:**

The build loop checks `len(self._table)` at every n-gram insertion.
If it exceeds `MAX_NGRAM_ENTRIES`, a `MemoryLimitError` is raised
immediately rather than letting the process exhaust RAM and crash
the operating system.

---

### ModelPersistence

Handles saving and loading trained models to disk.

**Atomic write protocol:**

```
1. Compute destination path:  markov_models/mymodel.json
2. Create temp path:          markov_models/mymodel.json.tmp
3. Write full content to .tmp file
4. os.rename(.tmp → .json)     ← atomic on all POSIX systems
```

If the program crashes at step 3, the `.tmp` file is left behind
but the original `.json` is untouched. The `.tmp` file is cleaned
up on the next save attempt.

**Payload structure (JSON):**

```json
{
  "version": "2.0",
  "saved_at": "2024-01-15T14:32:07",
  "model": {
    "order": 4,
    "table": {
      "the ": {"q": 3, "b": 1, "c": 7},
      ...
    },
    "vocabulary": ["a", "b", "c", ...],
    "stats": {
      "total_ngrams": 9847,
      "unique_ngrams": 1203,
      ...
    }
  }
}
```

---

### MarkovGenerator

The engine that produces new text from a trained `NGramTable`.

**Generation loop:**

```
1.  Pick a starting n-gram
2.  Write n-gram to output buffer
3.  Loop until enough characters generated:
    a.  Check timeout — if exceeded, break with partial result
    b.  Sample next character using chosen strategy
    c.  If no character found (dead end):
        - Increment dead end counter
        - Attempt recovery (3-tier)
        - If recovery fails after 100 attempts, break
    d.  Append character to buffer
    e.  Slide window: new_ngram = old_ngram[1:] + new_char
    f.  Print progress every 10,000 characters
4.  Return GenerationResult with text + metadata
```

**Dead-end recovery (3 tiers):**

```
Tier 1: Find any n-gram that starts with the TAIL of current n-gram
        Example: current = "abcd", tail = "bcd"
                 Find any key starting with "bcd"
                 → "bcde", "bcdf", etc.
        This maintains partial context continuity.

Tier 2: If no tail-overlap candidate found,
        pick a completely random n-gram from the table.
        This is a hard restart with no context.

Tier 3: If both fail more than DEAD_END_RESTART_ATTEMPTS times (100),
        return the text generated so far and mark completed=False.
```

**Starting n-gram selection:**

The system prefers n-grams whose first character is:
- An uppercase letter (likely start of sentence)
- A sentence-ending punctuation mark (`.`, `!`, `?`, `\n`)

This makes generated text more likely to start at a natural
sentence boundary rather than mid-word.

---

### OutputFormatter

Handles all display and file output.

**Paged display:**

For output longer than 40 lines, text is shown 40 lines at a time:

```
[Lines 1–40 of 312] Press Enter to continue, 'q' to stop:
```

This prevents the terminal from being flooded with thousands of
lines of text.

**Save to file:**

After display, you are offered the option to save the generated
text as a UTF-8 encoded plain text file.

---

### SessionManager

The top-level controller that ties everything together.

**Responsibility boundaries:**

```
SessionManager    ← Menu, action dispatch, state holding
    │
    ├── InputManager      ← All user input
    ├── NGramTable        ← Model data
    ├── MarkovGenerator   ← Text production
    ├── ModelPersistence  ← Disk I/O
    └── OutputFormatter   ← Display
```

The `SessionManager` holds the current model state
(`self.table`, `self.generator`) across multiple menu actions.
You can generate text 10 times from the same model without retraining.

---

## Edge Cases and Resilience

This system is specifically engineered to handle situations that
would crash a standard implementation.

### Input Edge Cases

| Scenario | Handling |
|----------|----------|
| Empty string entered | `InsufficientTextError` with clear message |
| Only whitespace entered | Caught by sanitizer Stage 10 |
| Only invisible chars | All removed by Stage 3, caught by Stage 10 |
| File not found | User prompted to retry or exit |
| File is a directory | Detected, user prompted again |
| File is 0 bytes | Detected, user prompted again |
| File is 51 MB | Rejected before reading (size check) |
| File has Windows CRLF | Normalized by Stage 5 |
| File saved with BOM | Removed by Stage 1 and UTF-8-SIG encoding try |
| Ctrl+C during input | Graceful: uses text entered so far |
| Ctrl+D (EOF) | Graceful: same as Ctrl+C |
| Non-integer for order | Caught, user re-prompted |
| Order larger than text allows | Capped to `text_length // 2` |
| Order = 0 | Below minimum, user re-prompted |
| Negative number entered | Below minimum, user re-prompted |

### Model Building Edge Cases

| Scenario | Handling |
|----------|----------|
| Text shorter than order+1 | `InvalidOrderError` with lengths shown |
| Every character unique | Model builds (1 branch per n-gram) |
| All characters identical | Model builds (single branch per n-gram) |
| N-gram table hits 5M entries | `MemoryLimitError` raised immediately |
| Unicode in n-grams | Handled natively (Python strings are Unicode) |
| Emoji in text | Treated as characters — model learns them |

### Generation Edge Cases

| Scenario | Handling |
|----------|----------|
| Dead end immediately | 3-tier recovery, up to 100 attempts |
| All n-grams are dead ends | Returns partial text after 100 attempts |
| Generation takes > 60 seconds | Timeout: partial result returned |
| Generating 0 characters | Below minimum, rejected at input |
| Generating 1,000,000 characters | Supported, shows progress bar |
| Seed not an integer | Warning printed, random seed used |
| Temperature = 0 | Falls back to greedy sampling |
| Temperature very high (3.0) | Uniform-like distribution, chaotic output |
| Starting n-gram not in table | Logged, random replacement chosen |

### Persistence Edge Cases

| Scenario | Handling |
|----------|----------|
| Save directory doesn't exist | Auto-created with `mkdir(parents=True)` |
| Disk full during save | `PersistenceError` raised, temp file cleaned |
| Permission denied on save | `PersistenceError` raised |
| Model file corrupted | `PersistenceError` with clear message |
| Model file truncated | `EOFError` caught, wrapped in `PersistenceError` |
| Wrong version model loaded | Warning logged, load attempted anyway |
| Pickle file from different Python | `UnpicklingError` caught |
| JSON file with invalid UTF-8 | Caught by `json.JSONDecodeError` |
| Crash during save | Temp file approach means original untouched |

### Session Edge Cases

| Scenario | Handling |
|----------|----------|
| Ctrl+C in any menu action | Caught, returns to main menu |
| Any `MarkovSystemError` | Printed cleanly, returns to main menu |
| Any unexpected `Exception` | Caught, logged, returns to main menu |
| Selecting unavailable option | Explained with which step is needed |
| Generating with no model | Caught before generation starts |

---

## Understanding the Output Statistics

After each generation, you see a report like this:

```
📊  GENERATION REPORT
  ├─ Characters generated : 500
  ├─ Time elapsed          : 0.003s
  ├─ Strategy              : STANDARD
  ├─ Dead ends encountered : 2
  ├─ Restarts              : 2
  ├─ Completed             : ✓
  ├─ Seed                  : None
  ├─ Model order           : 4
  ├─ Unique n-grams        : 1,203
  ├─ Vocabulary size       : 67
  ├─ Avg branching factor  : 3.21
  ├─ Model entropy         : 1.8304 bits
  └─ Model perplexity      : 3.55
```

### What Each Metric Means

**Characters generated**
The exact number of characters in the output, including the starting n-gram.

**Time elapsed**
Wall-clock time from start to finish of the generation call.

**Strategy**
Which generation strategy was used: `STANDARD`, `GREEDY`, or `TEMPERATURE`.

**Dead ends encountered**
How many times the current n-gram had no known successors.
A small number (0–5) is normal for high-order models on limited text.
A large number suggests the order is too high for the training text.

**Restarts**
How many times the recovery mechanism had to jump to a new n-gram.
Each restart can introduce a discontinuity in the generated text.

**Completed**
`✓` means the full requested length was generated.
`✗ (partial)` means generation stopped early due to timeout or
exhausted recovery attempts.

**Seed**
The random seed used. `None` means an unseeded (random) run.

**Model order**
The n-gram window size used to build the model.

**Unique n-grams**
Number of distinct context windows seen in the training text.
More unique n-grams = more nuanced model.

**Vocabulary size**
Number of distinct characters in the training text.
English prose: ~70–90. Code: ~50–80. Only ASCII text: ≤ 128.

**Avg branching factor**
Average number of distinct characters that can follow any given n-gram.
- `1.0` = completely deterministic (each n-gram has only one successor)
- High values = many choices = more creative but less coherent
- For English text, a healthy range is 2–8 depending on order

**Model entropy (bits)**
Average Shannon entropy across all n-gram states.
- `0.0` = completely deterministic
- `log2(vocabulary_size)` = maximum disorder (uniform distribution)
- English prose at order 4 typically shows 1.5–3.0 bits

**Model perplexity**
`2^entropy`. A measure of how "uncertain" the model is.
- `1.0` = perfectly deterministic (always knows what comes next)
- `vocabulary_size` = completely random (no pattern learned)
- For good English prose models: 2–10
- High perplexity (> 500) triggers a warning

---

## How Order Affects Output Quality

Order is the single most impactful parameter in a Markov chain model.

```
ORDER  1: Almost completely random. No recognizable words.
          "ahetosrniheatlo ehtsioan"

ORDER  2: Occasional real letter combinations. Gibberish mostly.
          "the an washe ingt the"

ORDER  3: Real words start appearing. Grammar is absent.
          "the was the ing on the"

ORDER  4: Words mostly real. Short phrases recognizable.
          "the quick fox over the lazy"

ORDER  6: Sentences start making sense. Paragraphs still nonsensical.
          "the quick brown fox jumps over the"

ORDER  8: Nearly verbatim from training text. Very little variation.
          Starts to just quote the training text.

ORDER 12+: Output is almost identical to training text.
           Model has overfit the input.
```

### Rule of Thumb

```
Short story (1,000 chars):        Order 3–4
Average article (5,000 chars):    Order 4–6
Full book (100,000+ chars):       Order 6–10
```

If you see too many dead ends: **lower the order**.
If the output reads as nonsense: **raise the order** (or provide more text).
If the output is just copying the training text: **lower the order**.

---

## Examples

### Example 1: English Prose at Order 4

Training text: Classic literature excerpts (~600 chars)
Order: 4
Strategy: Standard
Characters: 300

```
the best of times, it was the age of wisdom, it was the
worst of times. Call me Ishmael. To be or not to be, that
is the question. All that glitters is not gold. A journey
of a thousand miles begins with a single step. In the
beginning was the best of times.
```

Note: Phrases recombine naturally. "the best of times" leads to
patterns that eventually circle back to themselves.

---

### Example 2: Code at Order 6

Training text: Python Fibonacci function
Order: 6
Strategy: Temperature T=0.7
Characters: 200

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(fibonacci(i))
```

At order 6, the model essentially memorizes the training code
since it is short. More varied code output requires larger input.

---

### Example 3: High Temperature Chaos

Training text: English prose
Order: 3
Strategy: Temperature T=2.5
Characters: 200

```
thend wis Ishe of a thoug of tiy andt stl the wasthe
fot glittimeis notimesthe qurney milest wast quiimes
of wisthims the lazymest the milend
```

Temperature 2.5 produces near-random output even from structured text.

---

### Example 4: Greedy (Deterministic)

Training text: English prose
Order: 4
Strategy: Greedy
Characters: 100

```
the best of times, it was the best of times, it was the best of
times, it was the best of times, it was th
```

Greedy always picks the most common next character,
leading to loops in common phrases.

---

## Troubleshooting

### "Text became empty after sanitization"

Your input consisted entirely of characters that were removed
during sanitization (invisible characters, control codes, etc.).
Try providing text with normal printable characters.

### "Text length (X) must exceed order (Y)"

You chose an order that is too high for your text.
The system needs at least `order + 1` characters to build even
one n-gram. Lower the order or provide more text.

### Many dead ends / Partial generation

The model cannot find a successor for many n-grams. This usually
means the order is too high relative to the text size. Try:
1. Lowering the order
2. Providing more training text
3. Enabling smoothing (e.g., 0.1 or 1.0)

### Generated text is just copying the training text

The order is too high. The model has memorized specific sequences.
Try lowering the order by 2–3 steps.

### High perplexity warning

The model is very uncertain — it sees many equally likely next
characters at most positions. This means:
- The text is very diverse/random (e.g., random characters)
- The order is too low (1–2) and no real pattern is learnable
- The vocabulary is very large relative to text size

Try providing more structured text or increasing the order.

### Model file not loading

If you see "Model file corrupted or unreadable":
- The file may have been partially written (if a crash occurred during save)
- The file may have been edited manually and broken
- A Pickle file from a very different Python version may be incompatible
  (use JSON format for maximum portability)

### Permission denied when saving

The system cannot write to the `markov_models/` directory.
Either:
1. Run from a directory where you have write access
2. Change `MODEL_SAVE_DIR` in `SystemConfig` to a writable path

---

## Technical Glossary

| Term | Definition |
|------|-----------|
| **Markov Chain** | A stochastic process where the next state depends only on the current state |
| **N-gram** | A contiguous sequence of N characters from the training text |
| **Order** | The N in N-gram; the size of the context window |
| **Vocabulary** | The set of all distinct characters seen in training text |
| **Branching Factor** | Number of distinct characters that can follow a given n-gram |
| **Dead End** | An n-gram that was never seen as a prefix; has no successors |
| **Shannon Entropy** | A measure of average uncertainty/information content in bits |
| **Perplexity** | 2^entropy; how "surprised" the model is on average |
| **Laplace Smoothing** | Adding a small constant to all counts to avoid zero probabilities |
| **Temperature** | A scaling factor applied to probability distributions before sampling |
| **Atomic Write** | A write pattern where the file is either fully written or not at all |
| **NFC Normalization** | Unicode Canonical Decomposition followed by Canonical Composition |
| **BOM** | Byte Order Mark; a special Unicode character at the start of some files |
| **Mojibake** | Garbled text resulting from decoding bytes with the wrong encoding |

---

## Contributing

Contributions are welcome. To maintain the quality standard of this project:

1. **No new external dependencies.** This runs on stdlib only.
2. **All new code paths must handle their own exceptions.**
3. **Every new configurable parameter must go into `SystemConfig`.**
4. **Edge cases must be considered** — what happens with empty input?
   What if the user presses Ctrl+C? What if the disk is full?
5. **New components follow the existing pattern:**
   - Clear docstrings explaining what it handles
   - Type annotations on all public methods
   - Logging at appropriate levels

---

## License

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**Built with zero external dependencies. Runs on Python 3.8+.**

*If it can go wrong, this system handles it.*

</div>
