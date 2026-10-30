"""Rule-based verifier: reject a cache hit when the two questions differ in a
negation, a number, a unit, or a comparative/superlative.

This guard targets exactly the edits used to build the adversarial set, so its
results are a BEST CASE for those edit types. Entity swaps are not covered.
"""
import re

NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "without", "cannot"}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

# Unit words mapped to one canonical name (spelling variants and plurals merged).
UNITS = {}
for canon, variants in {
    "metre": ["metre", "metres", "meter", "meters"],
    "kilometre": ["kilometre", "kilometres", "kilometer", "kilometers", "km"],
    "centimetre": ["centimetre", "centimetres", "centimeter", "centimeters", "cm"],
    "millimetre": ["millimetre", "millimetres", "millimeter", "millimeters", "mm"],
    "mile": ["mile", "miles"], "yard": ["yard", "yards"], "foot": ["foot", "feet"],
    "inch": ["inch", "inches"], "pound": ["pound", "pounds", "lb", "lbs"],
    "ounce": ["ounce", "ounces"], "gram": ["gram", "grams"], "kilogram": ["kilogram", "kilograms", "kg"],
    "stone": ["stone", "stones"], "litre": ["litre", "litres", "liter", "liters"],
    "millilitre": ["millilitre", "millilitres", "milliliter", "milliliters"],
    "centilitre": ["centilitre", "centilitres", "centiliter", "centiliters"],
    "pint": ["pint", "pints"], "gallon": ["gallon", "gallons"],
    "celsius": ["celsius", "centigrade"], "fahrenheit": ["fahrenheit"], "kelvin": ["kelvin"],
    "second_t": [], "minute": ["minute", "minutes"], "hour": ["hour", "hours"],
    "day": ["day", "days"], "week": ["week", "weeks"], "acre": ["acre", "acres"],
    "hectare": ["hectare", "hectares"], "kilopascal": ["kilopascal", "kilopascals", "kpa"],
    "psi": ["psi"], "knot": ["knot", "knots"],
}.items():
    for v in variants:
        UNITS[v] = canon
# "seconds" (plural) is a time unit; singular "second" is treated as an ordinal above.
UNITS["seconds"] = "second_t"

# Comparative/superlative words mapped to a direction class, so synonyms
# (largest/biggest) agree but opposites (largest/smallest) do not.
COMPARATIVES = {}
for canon, words in {
    "big": ["largest", "biggest", "greatest", "larger", "bigger", "greater"],
    "small": ["smallest", "tiniest", "smaller", "least"],
    "most": ["most", "more"], "few": ["fewest", "fewer", "less"],
    "old": ["oldest", "older", "eldest", "elder"], "young": ["youngest", "younger"],
    "high": ["highest", "higher", "tallest", "taller"], "low": ["lowest", "lower", "shortest_h"],
    "long": ["longest", "longer"], "short": ["shortest", "shorter"],
    "deep": ["deepest", "deeper"], "shallow": ["shallowest", "shallower"],
    "hard": ["hardest", "harder"], "soft": ["softest", "softer"],
    "heavy": ["heaviest", "heavier"], "light": ["lightest", "lighter"],
    "near": ["closest", "nearest", "closer", "nearer"], "far": ["farthest", "furthest", "farther", "further"],
    "last": ["last", "final", "latest"], "early": ["earliest", "earlier"],
    "dense": ["densest", "denser"], "populous": ["populous"],
}.items():
    for w in words:
        COMPARATIVES[w] = canon

TOKEN_RE = re.compile(r"[a-z]+|\d[\d,]*\.?\d*")


def tokens(text):
    text = text.lower().replace("n't", " not").replace("’", "'")
    return TOKEN_RE.findall(text)


def features(text):
    toks = tokens(text)
    nums = set()
    for t in toks:
        if t[0].isdigit():
            nums.add(float(t.replace(",", "")))
        elif t in NUMBER_WORDS:
            nums.add(float(NUMBER_WORDS[t]))
    # "least populous" vs "most populous": the direction word decides.
    comps = {COMPARATIVES[t] for t in toks if t in COMPARATIVES}
    return {
        "negation": sum(t in NEGATIONS for t in toks),
        "numbers": nums,
        "units": {UNITS[t] for t in toks if t in UNITS},
        "comparatives": comps,
    }


def rule_guard(query, cached_question):
    """Return (accept, reason). Accept only if no guarded feature differs.

    Negation words are counted, not just detected, so a title such as
    "Much Ado About Nothing" in both questions does not hide an added "not".
    """
    a, b = features(query), features(cached_question)
    if a["negation"] != b["negation"]:
        return False, "negation differs"
    if a["numbers"] != b["numbers"]:
        return False, "numbers differ"
    if a["units"] != b["units"]:
        return False, "units differ"
    if a["comparatives"] != b["comparatives"]:
        return False, "comparatives differ"
    return True, ""
