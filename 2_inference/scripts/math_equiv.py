"""
Math equivalence checking for LaTeX answers.
Adapted from DeepSeek-AI/DeepSeek-Math (MIT License, Original authors: DeepSeek-AI (2024)),
according to Lewkowycz et al. (2022), Appendix D — exact string match first, then sympy simplification fallback.

With slightly changed symbolic comparison and added substitution terms.
"""

import logging, re, signal
from contextlib import contextmanager
from typing import Optional

import sympy
from sympy.parsing.latex import parse_latex

# Substitutions before removal (Lewkowycz et al. 2022, App. D).
_MATH_SUBSTITUTIONS: list[tuple[str, str]] = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""),
    (" ", ""), ("mbox", "text"), (",\\text{and}", ","),
    ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]

# Removed subwords (most adopted from Deepseek, with additions from qualitative analyses)
_MATH_REMOVED: list[str] = [
    # LaTeX markers
    "\\ldots", "\\dots",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3",
    "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!",
    "{,}", '"',
    # OCW units / words (Lewkowycz et al. 2022)
    "square", "ways", "integers", "dollars", "mph", "inches", "ft",
    "hours", "km", "units", "sue", "points", "feet", "minutes",
    "digits", "cents", "degrees", "cm", "gm", "pounds", "meters", "meals",
    "edges", "students", "childrentickets", "multiples",
    # Hendrycks MATH: time
    "seconds", "days", "weeks", "months", "years",
    # Hendrycks MATH: length / area / volume
    "miles", "yards", "millimeters", "kilometers", "hectares", "acres",
    "liters", "gallons", "ounces", "quarts", "pints", "cups",
    "milligrams", "kilograms", "grams", "tons",
    # Hendrycks MATH: money
    "pennies", "nickels", "dimes", "quarters", "coins",
    # Hendrycks MATH: people
    "people", "children", "boys", "girls", "men", "women",
    "teachers", "workers", "players", "teams", "field goals",
    # Hendrycks MATH: food / objects
    "cupcakes", "cookies", "muffins", "cakes", "pies", "brownies",
    "apples", "oranges", "bananas", "pears", "lemons", "grapes", "fruits",
    "marbles", "balls", "tiles", "blocks", "beads",
    "books", "cards", "boxes", "bags", "pens", "pencils", "chairs",
    "tables", "rooms", "shelves",
    # Hendrycks MATH: animals
    "cats", "dogs", "birds", "fish", "horses", "cows", "sheep",
    "rabbits", "frogs", "trees", "flowers", "plants",
    # Excess words
    "should be", "the", "would be", "won", "**", 
]

_DROP_REMOVED: list[str] = [
    # Excess words
    "should be", "the", "would be", "won", "**", 
]

@contextmanager
def _time_limit(seconds: int = 5):
    """SIGALRM-based timeout (Unix only)."""
    def _handler(signum, frame): raise TimeoutError
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try: yield
    finally: signal.alarm(0)


def _normalize_tex(expr: str) -> str:
    """Normalisation of Latex responses according to Lewkowycz et al. (2022)."""
    expr = expr.split("=")[-1]
    for before, after in _MATH_SUBSTITUTIONS: expr = expr.replace(before, after)
    for token in _MATH_REMOVED: expr = expr.replace(token, "")
    expr = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", expr)
    expr = re.sub(r"(\\text\{)(.*?)(\})", "\\2", expr)
    expr = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", expr)
    expr = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", expr)
    expr = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", expr)
    expr = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", expr)
    expr = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", expr)
    expr = expr.replace("$", "")
    if expr.replace(",", "").isdigit(): expr = expr.replace(",", "")
    return expr


def _parse_tex(text: str, time_limit: int = 5) -> Optional[sympy.Basic]:
    """Parse latex string to sympy expression. None on failure"""
    try:
        with _time_limit(time_limit):
            return parse_latex(text)
    except Exception as e:
        logging.debug(f"[MATH] parse failed for '{text}': {e}")
        return None


def _sympy_equiv(x1: sympy.Basic, x2: sympy.Basic, time_limit: int = 5) -> bool:
    """Return True if difference between x1 and x2 simplifies to zero."""
    try:
        with _time_limit(time_limit):
            return sympy.simplify(x1 - x2) == 0
    except Exception:
        return False


def is_math_equiv(pred: str, ref: str, time_limit: int = 5) -> bool:
    """
    Return True if pred and ref are mathematically equivalent LaTeX expressions.
    Normalises both strings then exact string match first, and sympy fallback second according to Lewkowycz et al. (2022).
    """
    pred_n = _normalize_tex(pred)
    ref_n  = _normalize_tex(ref)
    if pred_n == ref_n: return True
    parsed_ref = _parse_tex(ref_n, time_limit)
    if parsed_ref is None: return False
    return _sympy_equiv(_parse_tex(pred_n, time_limit), parsed_ref, time_limit)


def _normalize_drop(answer: str) -> str:
    """Normalize a DROP answer: numbers to canonical int/float, spans to lowercase."""
    answer = answer.strip()
    for token in _DROP_REMOVED:
        answer = answer.replace(token, "")
    try:
        num = float(answer.replace(",", "")) # ie. 1,000
        return str(int(num)) if num == int(num) else str(num)
    except ValueError:
        return answer.lower()


def drop_equiv(pred: str, ref: str) -> bool:
    """
    Equivalence check for DROP dataset answers (numeric and span types).
    If ref is bare number, extract first number from pred (ie. "42 yards", "there are 3 more") pre-comparison.
    Otherwise fallback to case-insensitive string match.
    """
    ref_n  = _normalize_drop(ref)
    pred_n = _normalize_drop(pred)
    if ref_n == pred_n: return True
    if re.fullmatch(r"-?\d+\.?\d*", ref_n):
        m = re.search(r"-?\d[\d,]*\.?\d*", pred)
        if m: return _normalize_drop(m.group()) == ref_n
    return False


# Maps equiv-key of HF_DATASETS config to equivalence fn.
_EQUIV_FNS: dict[str, object] = {
    "math":  is_math_equiv,
    "drop":  drop_equiv,
    "exact": None,
}