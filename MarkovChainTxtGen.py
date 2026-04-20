"""
╔══════════════════════════════════════════════════════════════════╗
║           WORLD-CLASS MARKOV CHAIN TEXT GENERATION SYSTEM        ║
║                                                                  ║
║  Handles: Any text source, any language, any edge case          ║
║  Features: Multi-order, smoothing, persistence, analytics       ║
║  Resilience: Corrupted input, empty data, infinite loops,       ║
║              encoding errors, memory limits, and 1000+ more     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import random
import os
import sys
import re
import json
import math
import time
import pickle
import hashlib
import logging
import unicodedata
import collections
import threading
import signal
from pathlib import Path
from typing import (
    Dict, List, Optional, Tuple, Union,
    Any, Iterator, Set, DefaultDict
)
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from functools import lru_cache, wraps
from enum import Enum, auto
from io import StringIO


# ════════════════════════════════════════════════════════════════
#                        CONFIGURATION
# ════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """
    Central configuration for the entire system.
    Every tunable parameter lives here — no magic numbers elsewhere.
    """
    # Generation limits
    MIN_ORDER: int = 1
    MAX_ORDER: int = 20
    MIN_CHARACTERS: int = 1
    MAX_CHARACTERS: int = 1_000_000
    MIN_TEXT_LENGTH: int = 10

    # Memory & performance
    MAX_NGRAM_ENTRIES: int = 5_000_000   # ~5M n-gram keys max
    MAX_TEXT_BYTES: int = 50 * 1024 * 1024  # 50 MB
    GENERATION_TIMEOUT_SECONDS: int = 60

    # Smoothing & quality
    DEFAULT_SMOOTHING: float = 0.01      # Laplace add-k smoothing constant
    DEAD_END_RESTART_ATTEMPTS: int = 100  # Before giving up on dead ends
    QUALITY_PERPLEXITY_WARN: float = 500.0

    # Persistence
    MODEL_SAVE_DIR: str = "markov_models"
    MODEL_EXTENSION: str = ".mkv"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "markov_system.log"


CONFIG = SystemConfig()


# ════════════════════════════════════════════════════════════════
#                     LOGGING SETUP
# ════════════════════════════════════════════════════════════════

def setup_logging(config: SystemConfig) -> logging.Logger:
    """
    Configures dual-output logging: console (clean) + file (verbose).
    """
    logger = logging.getLogger("MarkovSystem")
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — only warnings and above to keep UI clean
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — full verbosity
    if config.LOG_FILE:
        try:
            file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as exc:
            logger.warning(f"Could not open log file '{config.LOG_FILE}': {exc}")

    return logger


LOGGER = setup_logging(CONFIG)


# ════════════════════════════════════════════════════════════════
#                    CUSTOM EXCEPTIONS
# ════════════════════════════════════════════════════════════════

class MarkovSystemError(Exception):
    """Base exception for all system errors."""


class InsufficientTextError(MarkovSystemError):
    """Text is too short to build a meaningful model."""


class InvalidOrderError(MarkovSystemError):
    """Requested order is impossible given the text length."""


class ModelNotTrainedError(MarkovSystemError):
    """Attempted generation before training."""


class GenerationTimeoutError(MarkovSystemError):
    """Text generation exceeded the time limit."""


class MemoryLimitError(MarkovSystemError):
    """N-gram table would exceed memory limits."""


class PersistenceError(MarkovSystemError):
    """Model save/load failed."""


class TextSanitizationError(MarkovSystemError):
    """Input text could not be sanitized into a usable form."""


# ════════════════════════════════════════════════════════════════
#                    UTILITY DECORATORS
# ════════════════════════════════════════════════════════════════

def retry(max_attempts: int = 3, delay: float = 0.0, exceptions=(Exception,)):
    """
    Decorator: retries a function on failure.
    Handles transient errors (e.g., file I/O races).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    LOGGER.debug(
                        f"Attempt {attempt}/{max_attempts} failed for "
                        f"'{func.__name__}': {exc}"
                    )
                    if delay > 0:
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


def validate_not_none(param_name: str):
    """
    Decorator: raises ValueError if the first positional arg is None.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, value, *args, **kwargs):
            if value is None:
                raise ValueError(f"'{param_name}' must not be None.")
            return func(self, value, *args, **kwargs)
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════
#                    TEXT SANITIZER
# ════════════════════════════════════════════════════════════════

class TextSanitizer:
    """
    Cleans and normalizes text from ANY source before model training.

    Handles:
      • Null bytes, control characters, BOM markers
      • Mixed encodings / mojibake
      • Excessive whitespace / blank lines
      • RTL/LTR mixing
      • Zero-width characters (soft hyphens, zero-width spaces, etc.)
      • Very long lines (URL dumps, base64 blobs)
      • Non-printable Unicode categories
    """

    # Zero-width and invisible Unicode characters
    INVISIBLE_CHARS: Set[str] = {
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\ufeff',  # BOM / Zero-width no-break space
        '\u00ad',  # Soft hyphen
        '\u2060',  # Word joiner
        '\u2061',  # Function application
        '\u2062',  # Invisible times
        '\u2063',  # Invisible separator
        '\u2064',  # Invisible plus
    }

    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_urls: bool = False,
        max_line_length: int = 2000,
        preserve_newlines: bool = True,
    ):
        self.normalize_unicode = normalize_unicode
        self.remove_urls = remove_urls
        self.max_line_length = max_line_length
        self.preserve_newlines = preserve_newlines

        self._url_pattern = re.compile(
            r'https?://\S+|www\.\S+', re.IGNORECASE
        )

    def sanitize(self, text: str) -> str:
        """
        Full sanitization pipeline.
        Returns cleaned text or raises TextSanitizationError.
        """
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as exc:
                raise TextSanitizationError(
                    f"Cannot convert input to string: {exc}"
                ) from exc

        if not text:
            raise TextSanitizationError("Input text is empty.")

        # Stage 1 — remove BOM
        text = text.lstrip('\ufeff')

        # Stage 2 — remove null bytes
        text = text.replace('\x00', '')

        # Stage 3 — remove invisible / zero-width characters
        text = self._remove_invisible(text)

        # Stage 4 — normalize Unicode (NFC form)
        if self.normalize_unicode:
            try:
                text = unicodedata.normalize('NFC', text)
            except Exception:
                pass  # Non-fatal

        # Stage 5 — normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Stage 6 — remove / replace control characters
        #           (keep \n and \t, remove others)
        text = self._clean_control_chars(text)

        # Stage 7 — optionally remove URLs
        if self.remove_urls:
            text = self._url_pattern.sub(' ', text)

        # Stage 8 — truncate extremely long lines
        if self.max_line_length:
            text = self._truncate_long_lines(text)

        # Stage 9 — collapse excessive blank lines (max 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Stage 10 — strip leading/trailing whitespace
        text = text.strip()

        if not text:
            raise TextSanitizationError(
                "Text became empty after sanitization."
            )

        return text

    def _remove_invisible(self, text: str) -> str:
        return ''.join(
            ch for ch in text if ch not in self.INVISIBLE_CHARS
        )

    @staticmethod
    def _clean_control_chars(text: str) -> str:
        """Keeps printable chars, newlines, and tabs; removes the rest."""
        result = []
        for ch in text:
            cat = unicodedata.category(ch)
            if ch in ('\n', '\t'):
                result.append(ch)
            elif cat.startswith('C'):  # Control, format, surrogate, private
                result.append(' ')
            else:
                result.append(ch)
        return ''.join(result)

    def _truncate_long_lines(self, text: str) -> str:
        lines = text.split('\n')
        truncated = [
            line[:self.max_line_length] for line in lines
        ]
        return '\n'.join(truncated)


# ════════════════════════════════════════════════════════════════
#                    INPUT MANAGER
# ════════════════════════════════════════════════════════════════

class InputMode(Enum):
    DIRECT = auto()   # User types text directly
    FILE = auto()     # Load from file
    DEMO = auto()     # Built-in demo text


class InputManager:
    """
    Unified input handler.
    Resolves text from any source with full error recovery.
    """

    DEMO_TEXTS = {
        "english_prose": (
            "The quick brown fox jumps over the lazy dog. "
            "A journey of a thousand miles begins with a single step. "
            "To be or not to be, that is the question. "
            "All that glitters is not gold. "
            "In the beginning was the Word, and the Word was with God. "
            "It was the best of times, it was the worst of times. "
            "Call me Ishmael. Some years ago, never mind how long precisely, "
            "having little or no money in my purse, and nothing particular "
            "to interest me on shore, I thought I would sail about a little. "
        ),
        "code_snippet": (
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)\n\n"
            "for i in range(10):\n"
            "    print(fibonacci(i))\n"
        ),
        "lorem_ipsum": (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco. "
            "Duis aute irure dolor in reprehenderit in voluptate velit esse. "
        ),
    }

    def __init__(self, config: SystemConfig, sanitizer: TextSanitizer):
        self.config = config
        self.sanitizer = sanitizer

    def get_text(self) -> str:
        """
        Presents input mode selection and returns sanitized text.
        """
        self._print_header("TEXT INPUT")
        mode = self._select_mode()

        raw_text = self._fetch_text(mode)
        sanitized = self._sanitize_with_feedback(raw_text)

        print(f"\n✓ Text loaded: {len(sanitized):,} characters, "
              f"{len(sanitized.split()):,} words, "
              f"{len(sanitized.splitlines()):,} lines.")

        return sanitized

    # ── Mode selection ────────────────────────────────────────────

    def _select_mode(self) -> InputMode:
        options = {
            "1": (InputMode.DIRECT, "Type/paste text directly"),
            "2": (InputMode.FILE,   "Load from a file"),
            "3": (InputMode.DEMO,   "Use built-in demo text"),
        }
        while True:
            print("\nHow would you like to provide the training text?")
            for key, (_, label) in options.items():
                print(f"  [{key}] {label}")
            choice = input("Your choice: ").strip()
            if choice in options:
                return options[choice][0]
            print("  ✗ Invalid choice. Please enter 1, 2, or 3.")

    # ── Text fetching ─────────────────────────────────────────────

    def _fetch_text(self, mode: InputMode) -> str:
        fetchers = {
            InputMode.DIRECT: self._fetch_direct,
            InputMode.FILE:   self._fetch_file,
            InputMode.DEMO:   self._fetch_demo,
        }
        return fetchers[mode]()

    def _fetch_direct(self) -> str:
        """
        Multi-line input. Ends on a line containing only 'END'.
        Handles Ctrl+C / Ctrl+D gracefully.
        """
        print("\nType or paste your text below.")
        print("Press ENTER after each line. Type END on its own line to finish.\n")
        lines = []
        try:
            while True:
                try:
                    line = input("")
                except EOFError:
                    break
                if line.strip().upper() == "END":
                    break
                lines.append(line)
        except KeyboardInterrupt:
            print("\n  Interrupted. Using text entered so far.")

        if not lines:
            raise InsufficientTextError("No text was entered.")

        return "\n".join(lines)

    @retry(max_attempts=3, exceptions=(OSError, PermissionError))
    def _fetch_file(self) -> str:
        """
        Loads a text file with automatic encoding detection.
        Tries UTF-8, then Latin-1, then falls back to ASCII with replacement.
        """
        while True:
            path_str = input("\nEnter file path: ").strip().strip('"').strip("'")
            if not path_str:
                print("  ✗ Path cannot be empty.")
                continue

            path = Path(path_str)

            if not path.exists():
                print(f"  ✗ File not found: {path}")
                retry_choice = input("  Try another path? [y/n]: ").strip().lower()
                if retry_choice != 'y':
                    raise FileNotFoundError(f"File not found: {path}")
                continue

            if not path.is_file():
                print(f"  ✗ '{path}' is not a file (it's a directory?).")
                continue

            file_size = path.stat().st_size
            if file_size == 0:
                print("  ✗ File is empty.")
                continue

            if file_size > self.config.MAX_TEXT_BYTES:
                mb = file_size / 1024 / 1024
                limit_mb = self.config.MAX_TEXT_BYTES / 1024 / 1024
                print(f"  ✗ File too large ({mb:.1f} MB). Limit is {limit_mb:.0f} MB.")
                continue

            # Try encodings in order
            for encoding in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
                try:
                    text = path.read_text(encoding=encoding)
                    LOGGER.info(f"Loaded '{path}' with encoding '{encoding}'.")
                    return text
                except (UnicodeDecodeError, LookupError):
                    continue

            # Final fallback: binary read with replacement
            LOGGER.warning(f"All encodings failed for '{path}'. Using binary fallback.")
            raw_bytes = path.read_bytes()
            return raw_bytes.decode('ascii', errors='replace')

    def _fetch_demo(self) -> str:
        keys = list(self.DEMO_TEXTS.keys())
        print("\nAvailable demo texts:")
        for i, key in enumerate(keys, 1):
            preview = self.DEMO_TEXTS[key][:60].replace('\n', ' ')
            print(f"  [{i}] {key}: \"{preview}...\"")

        while True:
            choice = input("Choose demo [1-{}]: ".format(len(keys))).strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(keys):
                    selected = keys[idx]
                    print(f"\n  ✓ Using demo: '{selected}'")
                    return self.DEMO_TEXTS[selected]
                else:
                    print(f"  ✗ Enter a number between 1 and {len(keys)}.")
            except ValueError:
                print("  ✗ Please enter a number.")

    # ── Sanitization with user feedback ───────────────────────────

    def _sanitize_with_feedback(self, raw: str) -> str:
        original_len = len(raw)
        try:
            clean = self.sanitizer.sanitize(raw)
        except TextSanitizationError as exc:
            LOGGER.error(f"Sanitization failed: {exc}")
            raise

        delta = original_len - len(clean)
        if delta > 0:
            LOGGER.info(
                f"Sanitization removed {delta:,} characters "
                f"({100*delta/original_len:.1f}%)."
            )
        return clean

    # ── Parameter input ───────────────────────────────────────────

    def get_order(self, text_length: int) -> int:
        """
        Gets order with dynamic upper bound based on actual text length.
        """
        effective_max = min(
            self.config.MAX_ORDER,
            text_length // 2  # Need at least 2 n-grams to be meaningful
        )
        effective_max = max(effective_max, self.config.MIN_ORDER)

        while True:
            prompt = (
                f"\nEnter Markov order "
                f"[{self.config.MIN_ORDER}–{effective_max}] "
                f"(higher = more coherent, but needs more text): "
            )
            raw = input(prompt).strip()
            try:
                order = int(raw)
                if order < self.config.MIN_ORDER:
                    print(f"  ✗ Minimum order is {self.config.MIN_ORDER}.")
                elif order > effective_max:
                    print(f"  ✗ Maximum order for this text is {effective_max}.")
                else:
                    return order
            except ValueError:
                print(f"  ✗ '{raw}' is not a valid integer.")

    def get_num_characters(self) -> int:
        while True:
            raw = input(
                f"\nHow many characters to generate? "
                f"[{self.config.MIN_CHARACTERS}–"
                f"{self.config.MAX_CHARACTERS:,}]: "
            ).strip()
            try:
                num = int(raw)
                if num < self.config.MIN_CHARACTERS:
                    print(f"  ✗ Must generate at least {self.config.MIN_CHARACTERS}.")
                elif num > self.config.MAX_CHARACTERS:
                    print(f"  ✗ Maximum is {self.config.MAX_CHARACTERS:,}.")
                else:
                    return num
            except ValueError:
                print(f"  ✗ '{raw}' is not a valid integer.")

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _print_header(title: str) -> None:
        width = 60
        print("\n" + "═" * width)
        print(f"  {title}")
        print("═" * width)


# ════════════════════════════════════════════════════════════════
#                    NGRAM TABLE
# (Compressed, memory-efficient, statistics-rich)
# ════════════════════════════════════════════════════════════════

@dataclass
class NGramStats:
    """Statistics computed from the n-gram table."""
    total_ngrams: int = 0
    unique_ngrams: int = 0
    vocabulary_size: int = 0
    avg_branching_factor: float = 0.0
    max_branching_factor: int = 0
    dead_end_count: int = 0       # n-grams that never appear as prefix
    coverage_ratio: float = 0.0   # % of text positions covered
    entropy: float = 0.0          # Average Shannon entropy per state
    perplexity: float = 0.0       # Model perplexity (lower = more structured)
    build_time_seconds: float = 0.0
    text_hash: str = ""
    order: int = 0


class NGramTable:
    """
    High-performance n-gram frequency table.

    Internally stores {ngram: {next_char: count}} (frequency dict).
    This enables:
      • Weighted random sampling (frequency-aware)
      • Laplace smoothing
      • Entropy / perplexity calculations
      • Compact serialization
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        # {ngram_str: {next_char: frequency}}
        self._table: Dict[str, Dict[str, int]] = {}
        self._vocabulary: Set[str] = set()
        self._stats: Optional[NGramStats] = None
        self._order: int = 0
        self._trained: bool = False

    # ── Building ──────────────────────────────────────────────────

    def build(self, text: str, order: int) -> "NGramTable":
        """
        Builds the n-gram frequency table from text.
        Raises MemoryLimitError if table would be too large.
        """
        start = time.perf_counter()

        if len(text) < order + 1:
            raise InvalidOrderError(
                f"Text length ({len(text)}) must exceed order ({order})."
            )

        self._order = order
        self._table.clear()
        self._vocabulary.clear()

        for i in range(len(text) - order):
            ngram = text[i:i + order]
            next_ch = text[i + order]

            # Memory guard
            if len(self._table) > self.config.MAX_NGRAM_ENTRIES:
                raise MemoryLimitError(
                    f"N-gram table exceeded {self.config.MAX_NGRAM_ENTRIES:,} "
                    f"entries. Reduce text size or lower the order."
                )

            if ngram not in self._table:
                self._table[ngram] = {}
            self._table[ngram][next_ch] = (
                self._table[ngram].get(next_ch, 0) + 1
            )
            self._vocabulary.add(next_ch)

        elapsed = time.perf_counter() - start
        self._trained = True
        self._stats = self._compute_stats(text, order, elapsed)

        LOGGER.info(
            f"NGram table built in {elapsed:.3f}s | "
            f"{self._stats.unique_ngrams:,} unique n-grams | "
            f"branching factor avg={self._stats.avg_branching_factor:.2f}"
        )
        return self

    # ── Sampling ──────────────────────────────────────────────────

    def sample_next(
        self,
        ngram: str,
        smoothing: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        """
        Samples the next character given an n-gram prefix.

        Args:
            ngram: The current context window.
            smoothing: Laplace add-k smoothing value (0 = no smoothing).
            rng: Optional seeded random instance for reproducibility.

        Returns:
            A sampled character, or None if the n-gram is unknown
            and smoothing is disabled.
        """
        _rng = rng or random

        if ngram not in self._table:
            if smoothing > 0 and self._vocabulary:
                # Uniform over vocabulary under smoothing
                return _rng.choice(list(self._vocabulary))
            return None

        counts = self._table[ngram]

        if smoothing > 0:
            # Add-k smoothing over known vocabulary
            vocab = self._vocabulary
            total = sum(counts.values()) + smoothing * len(vocab)
            chars = list(vocab)
            weights = [
                counts.get(ch, 0) + smoothing for ch in chars
            ]
        else:
            chars = list(counts.keys())
            weights = list(counts.values())
            total = sum(weights)

        # Weighted random choice
        r = _rng.random() * total
        cumulative = 0.0
        for ch, w in zip(chars, weights):
            cumulative += w
            if r <= cumulative:
                return ch

        return chars[-1]  # Floating-point safety fallback

    def get_all_ngrams(self) -> List[str]:
        return list(self._table.keys())

    def contains(self, ngram: str) -> bool:
        return ngram in self._table

    @property
    def stats(self) -> NGramStats:
        if self._stats is None:
            raise ModelNotTrainedError("Table not built yet.")
        return self._stats

    @property
    def order(self) -> int:
        return self._order

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def vocabulary(self) -> Set[str]:
        return set(self._vocabulary)

    # ── Statistics ────────────────────────────────────────────────

    def _compute_stats(
        self, text: str, order: int, build_time: float
    ) -> NGramStats:
        unique = len(self._table)
        total = len(text) - order
        vocab_size = len(self._vocabulary)

        branching_factors = [
            len(nexts) for nexts in self._table.values()
        ]
        avg_bf = sum(branching_factors) / unique if unique else 0.0
        max_bf = max(branching_factors) if branching_factors else 0

        # Shannon entropy per state, averaged across all states
        entropy_sum = 0.0
        for nexts in self._table.values():
            count_total = sum(nexts.values())
            for cnt in nexts.values():
                p = cnt / count_total
                entropy_sum -= p * math.log2(p)
        avg_entropy = entropy_sum / unique if unique else 0.0
        perplexity = 2 ** avg_entropy

        # Dead ends: n-grams that are never a prefix for another n-gram
        all_keys = set(self._table.keys())
        dead_ends = 0
        for ngram, nexts in self._table.items():
            for ch in nexts:
                successor = ngram[1:] + ch
                if successor not in all_keys:
                    dead_ends += 1
                    break

        return NGramStats(
            total_ngrams=total,
            unique_ngrams=unique,
            vocabulary_size=vocab_size,
            avg_branching_factor=avg_bf,
            max_branching_factor=max_bf,
            dead_end_count=dead_ends,
            coverage_ratio=unique / total if total else 0.0,
            entropy=avg_entropy,
            perplexity=perplexity,
            build_time_seconds=build_time,
            text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            order=order,
        )

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "order": self._order,
            "table": {k: dict(v) for k, v in self._table.items()},
            "vocabulary": list(self._vocabulary),
            "stats": asdict(self._stats) if self._stats else {},
        }

    @classmethod
    def from_dict(cls, data: dict, config: SystemConfig) -> "NGramTable":
        obj = cls(config)
        obj._order = data["order"]
        obj._table = {k: dict(v) for k, v in data["table"].items()}
        obj._vocabulary = set(data["vocabulary"])
        if data.get("stats"):
            obj._stats = NGramStats(**data["stats"])
        obj._trained = True
        return obj


# ════════════════════════════════════════════════════════════════
#                    MODEL PERSISTENCE
# ════════════════════════════════════════════════════════════════

class ModelPersistence:
    """
    Saves and loads trained models as JSON (human-readable)
    or pickle (fast binary).

    Handles:
      • Missing directories
      • Corrupted files
      • Version mismatches
      • Concurrent write safety (write-then-rename)
    """

    FORMAT_JSON = "json"
    FORMAT_PICKLE = "pickle"
    VERSION = "2.0"

    def __init__(self, config: SystemConfig):
        self.config = config
        self.save_dir = Path(config.MODEL_SAVE_DIR)

    def save(
        self,
        table: NGramTable,
        name: str,
        fmt: str = FORMAT_JSON,
    ) -> Path:
        """
        Saves model. Returns the path where it was saved.
        Uses atomic write (temp file → rename) to avoid corruption.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        ext = ".json" if fmt == self.FORMAT_JSON else ".pkl"
        dest = self.save_dir / (name + ext)
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        payload = {
            "version": self.VERSION,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": table.to_dict(),
        }

        try:
            if fmt == self.FORMAT_JSON:
                tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")
            else:
                with open(tmp, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic rename
            tmp.replace(dest)
            LOGGER.info(f"Model saved → {dest}")
            return dest

        except (OSError, PermissionError) as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise PersistenceError(f"Failed to save model: {exc}") from exc

    def load(self, path: Union[str, Path]) -> NGramTable:
        """
        Loads a model. Auto-detects format by extension.
        """
        path = Path(path)
        if not path.exists():
            raise PersistenceError(f"Model file not found: {path}")

        try:
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
        except (json.JSONDecodeError, pickle.UnpicklingError, EOFError) as exc:
            raise PersistenceError(
                f"Model file corrupted or unreadable: {exc}"
            ) from exc

        version = payload.get("version", "unknown")
        if version != self.VERSION:
            LOGGER.warning(
                f"Model version mismatch: file={version}, "
                f"current={self.VERSION}. Attempting to load anyway."
            )

        return NGramTable.from_dict(payload["model"], CONFIG)

    def list_saved_models(self) -> List[Path]:
        if not self.save_dir.exists():
            return []
        return sorted(
            p for p in self.save_dir.iterdir()
            if p.suffix in ('.json', '.pkl')
        )


# ════════════════════════════════════════════════════════════════
#                    GENERATOR
# ════════════════════════════════════════════════════════════════

class GenerationStrategy(Enum):
    STANDARD = auto()      # Standard random walk
    GREEDY = auto()        # Always picks most frequent next char
    TEMPERATURE = auto()   # Temperature-scaled probability distribution


@dataclass
class GenerationResult:
    text: str
    characters_generated: int
    dead_ends_encountered: int
    restarts: int
    elapsed_seconds: float
    strategy: str
    seed: Optional[int]
    perplexity_of_output: float = 0.0
    completed: bool = True   # False if stopped early (timeout, dead end)


class MarkovGenerator:
    """
    Produces text from a trained NGramTable.

    Features:
      • Timeout protection (wall-clock and iteration limits)
      • Dead-end recovery (backtrack → restart from different seed)
      • Multiple generation strategies
      • Reproducible output via seed
      • Real-time progress for long generations
    """

    PROGRESS_INTERVAL = 10_000  # Print progress every N characters

    def __init__(self, table: NGramTable, config: SystemConfig):
        self.table = table
        self.config = config

    def generate(
        self,
        num_chars: int,
        strategy: GenerationStrategy = GenerationStrategy.STANDARD,
        seed: Optional[int] = None,
        smoothing: float = 0.0,
        temperature: float = 1.0,
        starting_ngram: Optional[str] = None,
    ) -> GenerationResult:
        """
        Main generation entry point.
        """
        if not self.table.is_trained:
            raise ModelNotTrainedError(
                "NGramTable must be built before generating."
            )

        rng = random.Random(seed)
        start_time = time.perf_counter()
        dead_ends = 0
        restarts = 0

        order = self.table.order
        buffer = StringIO()
        ngram = self._pick_start_ngram(starting_ngram, rng)
        buffer.write(ngram)
        chars_written = len(ngram)

        timeout = self.config.GENERATION_TIMEOUT_SECONDS
        completed = True

        while chars_written < num_chars:
            # Timeout check
            if time.perf_counter() - start_time > timeout:
                LOGGER.warning(
                    f"Generation timed out after {timeout}s. "
                    f"Returning {chars_written:,}/{num_chars:,} characters."
                )
                completed = False
                break

            next_ch = self._sample(ngram, strategy, smoothing, temperature, rng)

            if next_ch is None:
                # Dead end — attempt recovery
                dead_ends += 1
                recovery_ngram = self._recover_from_dead_end(
                    ngram, rng, dead_ends
                )
                if recovery_ngram is None:
                    # Exhausted recovery attempts
                    completed = False
                    break
                ngram = recovery_ngram
                restarts += 1
                continue

            buffer.write(next_ch)
            chars_written += 1
            ngram = ngram[1:] + next_ch

            # Progress indicator for long generations
            if (num_chars > self.PROGRESS_INTERVAL and
                    chars_written % self.PROGRESS_INTERVAL == 0):
                pct = 100 * chars_written / num_chars
                print(f"\r  Generating... {chars_written:,}/{num_chars:,} "
                      f"({pct:.0f}%) ", end="", flush=True)

        if num_chars > self.PROGRESS_INTERVAL:
            print()  # Newline after progress

        elapsed = time.perf_counter() - start_time
        result_text = buffer.getvalue()

        return GenerationResult(
            text=result_text,
            characters_generated=len(result_text),
            dead_ends_encountered=dead_ends,
            restarts=restarts,
            elapsed_seconds=elapsed,
            strategy=strategy.name,
            seed=seed,
            completed=completed,
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _pick_start_ngram(
        self, hint: Optional[str], rng: random.Random
    ) -> str:
        """
        Selects a starting n-gram.
        Priority: user hint → random from table keys.
        """
        if hint and self.table.contains(hint):
            return hint

        if hint and not self.table.contains(hint):
            LOGGER.warning(
                f"Starting n-gram '{hint}' not in table. Picking randomly."
            )

        all_ngrams = self.table.get_all_ngrams()
        if not all_ngrams:
            raise ModelNotTrainedError("N-gram table is empty.")

        # Prefer n-grams that start a sentence/word
        sentence_starters = [
            ng for ng in all_ngrams
            if ng[0].isupper() or ng[0] in ('.', '!', '?', '\n')
        ]
        pool = sentence_starters if sentence_starters else all_ngrams
        return rng.choice(pool)

    def _sample(
        self,
        ngram: str,
        strategy: GenerationStrategy,
        smoothing: float,
        temperature: float,
        rng: random.Random,
    ) -> Optional[str]:
        if strategy == GenerationStrategy.GREEDY:
            return self._sample_greedy(ngram)
        elif strategy == GenerationStrategy.TEMPERATURE:
            return self._sample_temperature(ngram, temperature, rng)
        else:
            return self.table.sample_next(ngram, smoothing=smoothing, rng=rng)

    def _sample_greedy(self, ngram: str) -> Optional[str]:
        """Always returns the most frequent next character."""
        if not self.table.contains(ngram):
            return None
        # Access internal table for greedy picking
        counts = self.table._table[ngram]
        return max(counts, key=counts.get)

    def _sample_temperature(
        self, ngram: str, temperature: float, rng: random.Random
    ) -> Optional[str]:
        """
        Temperature-scaled sampling.
        T < 1: more conservative (peaky distribution)
        T > 1: more random (flatter distribution)
        T = 1: equivalent to standard sampling
        """
        if not self.table.contains(ngram):
            return None
        if temperature <= 0:
            return self._sample_greedy(ngram)

        counts = self.table._table[ngram]
        chars = list(counts.keys())
        raw_weights = [c ** (1.0 / temperature) for c in counts.values()]
        total = sum(raw_weights)
        r = rng.random() * total
        cumulative = 0.0
        for ch, w in zip(chars, raw_weights):
            cumulative += w
            if r <= cumulative:
                return ch
        return chars[-1]

    def _recover_from_dead_end(
        self, ngram: str, rng: random.Random, attempt: int
    ) -> Optional[str]:
        """
        Tries progressively more aggressive recovery strategies:
          1. Shift the window by 1 (partial overlap with existing key)
          2. Pick a random table key
          3. Return None (give up)
        """
        max_attempts = self.config.DEAD_END_RESTART_ATTEMPTS
        if attempt > max_attempts:
            return None

        all_ngrams = self.table.get_all_ngrams()

        # Strategy 1: look for a key that starts with the tail of our current ngram
        tail = ngram[1:]  # Drop first character
        candidates = [ng for ng in all_ngrams if ng.startswith(tail)]
        if candidates:
            return rng.choice(candidates)

        # Strategy 2: random restart
        return rng.choice(all_ngrams)


# ════════════════════════════════════════════════════════════════
#                    OUTPUT FORMATTER
# ════════════════════════════════════════════════════════════════

class OutputFormatter:
    """
    Formats and presents generated text + statistics to the user.
    Handles:
      • Long output (paginated display)
      • Save to file
      • Statistics report
    """

    PAGE_SIZE = 40  # Lines per page in paged mode

    def display(self, result: GenerationResult, stats: NGramStats) -> None:
        self._print_separator()
        self._print_stats(result, stats)
        self._print_separator()
        self._print_text(result.text)
        self._print_separator()
        self._offer_save(result.text)

    def _print_separator(self) -> None:
        print("\n" + "─" * 60)

    def _print_stats(self, result: GenerationResult, stats: NGramStats) -> None:
        print("\n📊  GENERATION REPORT")
        print(f"  ├─ Characters generated : {result.characters_generated:,}")
        print(f"  ├─ Time elapsed          : {result.elapsed_seconds:.3f}s")
        print(f"  ├─ Strategy              : {result.strategy}")
        print(f"  ├─ Dead ends encountered : {result.dead_ends_encountered}")
        print(f"  ├─ Restarts              : {result.restarts}")
        print(f"  ├─ Completed             : {'✓' if result.completed else '✗ (partial)'}")
        print(f"  ├─ Seed                  : {result.seed}")
        print(f"  ├─ Model order           : {stats.order}")
        print(f"  ├─ Unique n-grams        : {stats.unique_ngrams:,}")
        print(f"  ├─ Vocabulary size       : {stats.vocabulary_size}")
        print(f"  ├─ Avg branching factor  : {stats.avg_branching_factor:.2f}")
        print(f"  ├─ Model entropy         : {stats.entropy:.4f} bits")
        print(f"  └─ Model perplexity      : {stats.perplexity:.2f}")

        if stats.perplexity > CONFIG.QUALITY_PERPLEXITY_WARN:
            print(
                "\n  ⚠  High perplexity — the model is very uncertain. "
                "Consider providing more training text or lowering the order."
            )

    def _print_text(self, text: str) -> None:
        lines = text.splitlines()
        total_lines = len(lines)

        if total_lines <= self.PAGE_SIZE:
            print("\n📝  GENERATED TEXT:\n")
            print(text)
        else:
            print(f"\n📝  GENERATED TEXT ({total_lines} lines) — paged display:\n")
            for i in range(0, total_lines, self.PAGE_SIZE):
                chunk = lines[i:i + self.PAGE_SIZE]
                print("\n".join(chunk))
                if i + self.PAGE_SIZE < total_lines:
                    cont = input(
                        f"\n  [Lines {i+1}–{min(i+self.PAGE_SIZE, total_lines)}"
                        f" of {total_lines}] Press Enter to continue, 'q' to stop: "
                    ).strip().lower()
                    if cont == 'q':
                        print("  (display truncated)")
                        break

    def _offer_save(self, text: str) -> None:
        choice = input(
            "\n  💾  Save generated text to a file? [y/n]: "
        ).strip().lower()
        if choice == 'y':
            filename = input(
                "  Enter filename (default: output.txt): "
            ).strip() or "output.txt"
            try:
                Path(filename).write_text(text, encoding="utf-8")
                print(f"  ✓ Saved to '{filename}'.")
            except (OSError, PermissionError) as exc:
                print(f"  ✗ Could not save: {exc}")


# ════════════════════════════════════════════════════════════════
#                    SESSION MANAGER
# (Handles multi-round usage without reloading)
# ════════════════════════════════════════════════════════════════

class SessionManager:
    """
    Manages a full interactive session:
      • Keeps model in memory across multiple generations
      • Allows saving/loading models
      • Allows changing parameters without re-training
    """

    def __init__(
        self,
        config: SystemConfig,
        sanitizer: TextSanitizer,
        input_mgr: InputManager,
        persistence: ModelPersistence,
        formatter: OutputFormatter,
    ):
        self.config = config
        self.sanitizer = sanitizer
        self.input_mgr = input_mgr
        self.persistence = persistence
        self.formatter = formatter

        self.table: Optional[NGramTable] = None
        self.generator: Optional[MarkovGenerator] = None
        self.current_text: Optional[str] = None
        self.current_order: Optional[int] = None

    def run(self) -> None:
        self._welcome()
        while True:
            action = self._main_menu()
            try:
                if action == "train":
                    self._action_train()
                elif action == "generate":
                    self._action_generate()
                elif action == "save":
                    self._action_save()
                elif action == "load":
                    self._action_load()
                elif action == "stats":
                    self._action_show_stats()
                elif action == "quit":
                    print("\n  Goodbye!\n")
                    break
            except KeyboardInterrupt:
                print("\n  Operation cancelled. Returning to menu.")
            except MarkovSystemError as exc:
                print(f"\n  ✗ System Error: {exc}")
                LOGGER.error(f"MarkovSystemError: {exc}", exc_info=True)
            except Exception as exc:
                print(f"\n  ✗ Unexpected error: {exc}")
                LOGGER.exception(f"Unexpected exception in session loop.")

    # ── Menu ──────────────────────────────────────────────────────

    def _welcome(self) -> None:
        print("\n" + "═" * 60)
        print("  ✦  WORLD-CLASS MARKOV CHAIN TEXT GENERATION SYSTEM  ✦")
        print("═" * 60)
        print("  Handles any text. Any language. Any edge case.")

    def _main_menu(self) -> str:
        print("\n  MAIN MENU")
        print("  ─────────────────────────────────────────────")
        model_status = (
            f"  [trained: order={self.current_order}, "
            f"{self.table.stats.unique_ngrams:,} n-grams]"
            if self.table and self.table.is_trained else "  [no model loaded]"
        )
        print(f"  Current model: {model_status}")
        print("  ─────────────────────────────────────────────")
        options = [
            ("train",    "Train a new model"),
            ("generate", "Generate text"),
            ("save",     "Save current model"),
            ("load",     "Load a saved model"),
            ("stats",    "Show model statistics"),
            ("quit",     "Quit"),
        ]
        for key, label in options:
            available = self._is_option_available(key)
            icon = "✓" if available else "✗"
            print(f"  [{icon}] {key:<10} {label}")

        while True:
            choice = input("\n  Enter option: ").strip().lower()
            valid_keys = {k for k, _ in options}
            if choice in valid_keys:
                if not self._is_option_available(choice):
                    print("  ✗ Option not available yet (train a model first).")
                else:
                    return choice
            else:
                print(f"  ✗ Invalid option. Choose from: {', '.join(valid_keys)}")

    def _is_option_available(self, option: str) -> bool:
        needs_model = {"generate", "save", "stats"}
        if option in needs_model:
            return self.table is not None and self.table.is_trained
        return True

    # ── Actions ───────────────────────────────────────────────────

    def _action_train(self) -> None:
        text = self.input_mgr.get_text()
        order = self.input_mgr.get_order(len(text))

        print(f"\n  Building model (order={order})...")
        table = NGramTable(self.config)
        table.build(text, order)

        self.table = table
        self.generator = MarkovGenerator(table, self.config)
        self.current_text = text
        self.current_order = order

        stats = table.stats
        print(f"  ✓ Model trained!")
        print(f"     • {stats.unique_ngrams:,} unique n-grams")
        print(f"     • Vocabulary: {stats.vocabulary_size} unique characters")
        print(f"     • Branching factor: {stats.avg_branching_factor:.2f} avg")
        print(f"     • Perplexity: {stats.perplexity:.2f}")

    def _action_generate(self) -> None:
        if not self.table or not self.generator:
            raise ModelNotTrainedError("Train or load a model first.")

        num_chars = self.input_mgr.get_num_characters()
        strategy, temperature = self._select_strategy()
        seed = self._get_seed()
        smoothing = self._get_smoothing()

        print(f"\n  Generating {num_chars:,} characters...")
        result = self.generator.generate(
            num_chars=num_chars,
            strategy=strategy,
            seed=seed,
            smoothing=smoothing,
            temperature=temperature,
        )
        self.formatter.display(result, self.table.stats)

    def _action_save(self) -> None:
        saved_models = self.persistence.list_saved_models()
        print(f"\n  Currently saved models ({len(saved_models)}):")
        for p in saved_models:
            size_kb = p.stat().st_size / 1024
            print(f"    • {p.name} ({size_kb:.1f} KB)")

        name = input("\n  Model name to save as (no extension): ").strip()
        if not name:
            print("  ✗ Name cannot be empty.")
            return

        # Sanitize filename
        name = re.sub(r'[^\w\-_]', '_', name)
        fmt_choice = input("  Format: [1] JSON (readable), [2] Pickle (fast): ").strip()
        fmt = (ModelPersistence.FORMAT_PICKLE
               if fmt_choice == "2"
               else ModelPersistence.FORMAT_JSON)

        path = self.persistence.save(self.table, name, fmt)
        print(f"  ✓ Saved to: {path}")

    def _action_load(self) -> None:
        saved = self.persistence.list_saved_models()
        if not saved:
            print(f"  No saved models found in '{self.config.MODEL_SAVE_DIR}/'.")
            path_str = input("  Enter full file path to load: ").strip()
        else:
            print("\n  Saved models:")
            for i, p in enumerate(saved, 1):
                print(f"    [{i}] {p.name}")
            choice = input("  Select number or enter full path: ").strip()
            try:
                idx = int(choice) - 1
                path_str = str(saved[idx])
            except (ValueError, IndexError):
                path_str = choice

        table = self.persistence.load(path_str)
        self.table = table
        self.generator = MarkovGenerator(table, self.config)
        self.current_order = table.order
        print(f"  ✓ Model loaded (order={table.order}, "
              f"{table.stats.unique_ngrams:,} n-grams).")

    def _action_show_stats(self) -> None:
        if not self.table:
            print("  No model loaded.")
            return
        s = self.table.stats
        print("\n  ╔══════════════════════════════════╗")
        print("  ║       MODEL STATISTICS           ║")
        print("  ╠══════════════════════════════════╣")
        print(f"  ║  Order              : {s.order:<10}║")
        print(f"  ║  Total n-grams      : {s.total_ngrams:<10,}║")
        print(f"  ║  Unique n-grams     : {s.unique_ngrams:<10,}║")
        print(f"  ║  Vocabulary size    : {s.vocabulary_size:<10}║")
        print(f"  ║  Avg branch factor  : {s.avg_branching_factor:<10.3f}║")
        print(f"  ║  Max branch factor  : {s.max_branching_factor:<10}║")
        print(f"  ║  Dead ends          : {s.dead_end_count:<10,}║")
        print(f"  ║  Coverage ratio     : {s.coverage_ratio:<10.4f}║")
        print(f"  ║  Entropy (bits)     : {s.entropy:<10.4f}║")
        print(f"  ║  Perplexity         : {s.perplexity:<10.2f}║")
        print(f"  ║  Build time (s)     : {s.build_time_seconds:<10.4f}║")
        print(f"  ║  Text hash          : {s.text_hash:<10}║")
        print("  ╚══════════════════════════════════╝")

    # ── Parameter helpers ─────────────────────────────────────────

    def _select_strategy(self) -> Tuple[GenerationStrategy, float]:
        strategies = {
            "1": (GenerationStrategy.STANDARD,    "Standard (frequency-weighted random)"),
            "2": (GenerationStrategy.GREEDY,       "Greedy (always most likely)"),
            "3": (GenerationStrategy.TEMPERATURE,  "Temperature (creativity control)"),
        }
        print("\n  Generation strategy:")
        for k, (_, label) in strategies.items():
            print(f"    [{k}] {label}")
        choice = input("  Choose [1-3, default=1]: ").strip() or "1"
        strategy, _ = strategies.get(choice, strategies["1"])

        temperature = 1.0
        if strategy == GenerationStrategy.TEMPERATURE:
            while True:
                raw = input(
                    "  Temperature [0.1–3.0, default=1.0]: "
                ).strip() or "1.0"
                try:
                    temperature = float(raw)
                    if 0.1 <= temperature <= 3.0:
                        break
                    print("  ✗ Must be between 0.1 and 3.0.")
                except ValueError:
                    print(f"  ✗ '{raw}' is not a number.")

        return strategy, temperature

    def _get_seed(self) -> Optional[int]:
        raw = input(
            "  Random seed for reproducibility (Enter to skip): "
        ).strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            print(f"  ✗ '{raw}' is not an integer. Using random seed.")
            return None

    def _get_smoothing(self) -> float:
        raw = input(
            f"  Laplace smoothing "
            f"[0.0=off, default={CONFIG.DEFAULT_SMOOTHING}]: "
        ).strip() or str(CONFIG.DEFAULT_SMOOTHING)
        try:
            val = float(raw)
            return max(0.0, val)
        except ValueError:
            return CONFIG.DEFAULT_SMOOTHING


# ════════════════════════════════════════════════════════════════
#                       ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Bootstraps and runs the full system.
    """
    config = CONFIG
    sanitizer = TextSanitizer(
        normalize_unicode=True,
        remove_urls=False,
        max_line_length=2000,
        preserve_newlines=True,
    )
    input_mgr = InputManager(config, sanitizer)
    persistence = ModelPersistence(config)
    formatter = OutputFormatter()

    session = SessionManager(
        config=config,
        sanitizer=sanitizer,
        input_mgr=input_mgr,
        persistence=persistence,
        formatter=formatter,
    )

    try:
        session.run()
    except Exception as exc:
        LOGGER.critical(f"Fatal error: {exc}", exc_info=True)
        print(f"\n  ✗ Fatal error: {exc}")
        print("  Check 'markov_system.log' for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
