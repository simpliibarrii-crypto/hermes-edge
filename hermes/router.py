import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

INTENT_CHAT = "chat"
INTENT_REASONING = "reasoning"
INTENT_TOOLS = "tools"


@dataclass
class IntentRule:
    pattern: str
    intent: str
    confidence: float


# Single list sorted by descending confidence.
# Highest-confidence rules are checked first so the first match wins.
INTENT_RULES: list[IntentRule] = [
    # ── Reasoning (deep thought, multi-step, logic) ────────────────
    IntentRule(r'\b(prove|reason|explain why|why does|why is|how does|logic|proof)', INTENT_REASONING, 0.96),
    IntentRule(r'\b(implement |write |code |function |program|algorithm|debug|syntax|compile)', INTENT_REASONING, 0.88),
    IntentRule(r'\b(compare|contrast|analyze|evaluate|what if|step by step|break down)', INTENT_REASONING, 0.78),
    IntentRule(r'\b(brainstorm|hypothesize|deduce|infer|synthesize|diagnose|troubleshoot)', INTENT_REASONING, 0.75),
    IntentRule(r'\b(think|deep|explain|why|how (do|does|did|would|can|could|should|will|may|might)|would |should |could |might )', INTENT_REASONING, 0.60),
    IntentRule(r'\b(what is the (meaning|purpose|point|reason) of)', INTENT_REASONING, 0.55),

    # ── Tools: Math & Calculation (highest confidence) ─────────────
    IntentRule(r'\b(calc|compute|math|add |sub |mul|div|sin|cos|tan|sqrt |factorial|solve )', INTENT_TOOLS, 0.95),
    IntentRule(r'\d+\s*[-+*/^%]\s*\d+', INTENT_TOOLS, 0.93),  # explicit expressions like 2+2

    # ── Tools: Weather & Environment ───────────────────────────────
    IntentRule(r'\b(weather|temperature|forecast|humidity|wind speed|barometer|precipitation|pressure|rain|snow|storm)', INTENT_TOOLS, 0.94),

    # ── Tools: Time, Date, Alarms ─────────────────────────────────
    IntentRule(r'\b(current time|what time|timezone|what date|todays date|set (timer|alarm|reminder))', INTENT_TOOLS, 0.93),

    # ── Tools: Web Search ─────────────────────────────────────────
    IntentRule(r'\b(web |search |find |look up|google |who is |define )', INTENT_TOOLS, 0.92),

    # ── Tools: Geography & Demographics ───────────────────────────
    IntentRule(r'\b(population|area |capital |currency |president|leader|language (spoken|official))', INTENT_TOOLS, 0.90),

    # ── Tools: Finance & Economics ────────────────────────────────
    IntentRule(r'\b(price |cost |salary |gdp |inflation|unemployment|stock |market |interest rate|exchange rate|bitcoin|crypto)', INTENT_TOOLS, 0.88),

    # ── Tools: What-is questions for real-time data ────────────────
    IntentRule(r'\b(what is( the)? (weather|time|date|population|capital|currency|price|distance|temperature))', INTENT_TOOLS, 0.86),

    # ── Tools: Conversions, Translations, Reminders ───────────────
    IntentRule(r'\b(convert |translate |summarize |remind |timer |alarm |schedule)', INTENT_TOOLS, 0.82),

    # ── Tools: Units & Measurements ───────────────────────────────
    IntentRule(r'\b(miles|kilometers|inches|centimeters|pounds|kilograms|gallons|liters|celsius|fahrenheit)', INTENT_TOOLS, 0.80),

    # ── Tools: News & Current Events ──────────────────────────────
    IntentRule(r'\b(news|headlines|latest |breaking |update on|what.s happening|current events)', INTENT_TOOLS, 0.79),

    # ── Tools: Directions & Navigation ────────────────────────────
    IntentRule(r'\b(directions|distance between|how far|route |map |navigate|address of|location of|where is)', INTENT_TOOLS, 0.77),

    # ── Chat: Greetings & Social ──────────────────────────────────
    IntentRule(r'\b(hello|hi |hey |good morning|good evening|what.s up|how are you|nice to|pleased)', INTENT_CHAT, 0.50),

    # ── Chat: Opinions & Preferences ──────────────────────────────
    IntentRule(r'\b(i think|i feel|i like|i prefer|i recommend|in my opinion|personally)', INTENT_CHAT, 0.45),

    # ── Chat: Simple Questions (low confidence catch-all) ──────────
    IntentRule(r'\b(who are you|what can you do|tell me about|help |what is your)', INTENT_CHAT, 0.40),
]


@dataclass
class RoutingResult:
    intent: str = INTENT_CHAT
    confidence: float = 0.0
    matched_pattern: str = ""


def classify(text: str) -> RoutingResult:
    """Classify intent in ~5μs. Zero model inference, pure keyword+regex.

    Rules are sorted by descending confidence so the first match wins.
    Returns:
        RoutingResult with intent, confidence, and matched pattern.
    """
    text_lower = text.lower()

    for rule in INTENT_RULES:
        if re.search(rule.pattern, text_lower):
            return RoutingResult(
                intent=rule.intent,
                confidence=rule.confidence,
                matched_pattern=rule.pattern,
            )

    return RoutingResult(intent=INTENT_CHAT, confidence=0.0)


def get_intent(text: str) -> str:
    """Convenience: returns just the intent string."""
    return classify(text).intent
