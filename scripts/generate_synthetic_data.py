#!/usr/bin/env python3
"""
Generate large-scale synthetic persuasion datasets using LLM-powered generation.

Target: 200K+ examples across 8 languages and 18 techniques.

Strategy:
  1. Template-based generation (fast, high-volume)
  2. LLM-augmented generation (GPT-4o-mini for diverse, realistic examples)
  3. Back-translation augmentation (expand language coverage)
  4. Paraphrase mining (create variants of seed examples)

Usage:
    python scripts/generate_synthetic_data.py --target-size 200000 --output data/synthetic/
    python scripts/generate_synthetic_data.py --target-size 5000 --mode templates-only
    python scripts/generate_synthetic_data.py --target-size 50000 --languages en,fr,es,de
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.data.collector import TECHNIQUE_LABELS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = ["en", "fr", "ar", "es", "de", "zh", "hi"]

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "ar": "Arabic", "es": "Spanish",
    "de": "German", "zh": "Chinese (Simplified)", "hi": "Hindi",
}

# Distribution weights for technique sampling (some more common than others)
TECHNIQUE_WEIGHTS = {
    "loaded_language": 1.5, "appeal_to_fear": 1.3, "appeal_to_emotion": 1.3,
    "exaggeration": 1.2, "name_calling": 1.1, "bandwagon": 1.0,
    "appeal_to_authority": 1.0, "false_dilemma": 0.9, "doubt": 0.9,
    "whataboutism": 0.8, "ad_hominem": 0.8, "causal_oversimplification": 0.8,
    "red_herring": 0.7, "straw_man": 0.7, "slogans": 0.7,
    "flag_waving": 0.6, "repetition": 0.6, "thought_terminating_cliche": 0.5,
}

# Domain contexts for diversity
DOMAINS = [
    "politics", "health", "climate", "economy", "technology",
    "education", "immigration", "military", "religion", "sports",
    "food_industry", "pharmaceuticals", "social_media", "energy",
    "criminal_justice", "housing", "labor", "international_relations",
]

TONES = [
    "aggressive", "subtle", "emotional", "authoritative", "casual",
    "formal", "urgent", "sarcastic", "populist", "academic",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SyntheticExample:
    """A single synthetic data point."""
    text: str
    language: str
    techniques: list[str]
    is_persuasive: bool
    domain: str = ""
    tone: str = ""
    source: str = "synthetic"
    generation_method: str = "template"
    difficulty: str = "medium"  # easy, medium, hard
    spans: list[dict] = field(default_factory=list)  # span-level annotations

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def uid(self) -> str:
        return hashlib.md5(self.text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Template-based generation (no API needed)
# ---------------------------------------------------------------------------

PERSUASIVE_TEMPLATES = {
    "en": {
        "appeal_to_fear": [
            "If we don't {action} immediately, {catastrophe} will be inevitable. Our {group} will suffer the consequences for generations.",
            "The threat of {threat} is closer than you think. Without {solution}, we face {disaster}. Time is running out.",
            "Wake up! {enemy} is already {bad_action}. By the time you realize the danger, it will be too late to {good_action}.",
            "Every day we delay {action}, {number} more {victims} are put at risk. How many must {suffer} before we act?",
        ],
        "loaded_language": [
            "This {negative_adj} policy is nothing but a {negative_noun} designed to {bad_goal} while honest {group} {suffer}.",
            "The {adjective} forces behind this {noun} have {adverb} {verb} everything we hold dear. It's a {loaded_term} against {victim}.",
            "What we're witnessing is nothing short of {extreme_noun} — a {adjective} assault on {value} by {enemy}.",
        ],
        "bandwagon": [
            "Everyone knows that {claim}. The vast majority of {group} already {action}. Don't be left behind.",
            "Millions of people have already {action}. Join the movement — {percentage}% of {group} agree that {claim}.",
            "Ask anyone on the street and they'll tell you: {claim}. It's common sense that {group} should {action}.",
        ],
        "appeal_to_authority": [
            "According to {authority}, this is absolutely {adjective}. With their {credentials}, how can anyone disagree?",
            "Dr. {name} from {institution} — a world-renowned {title} — has confirmed that {claim}. Case closed.",
            "Leading {experts} at {institution} unanimously agree: {claim}. The science is settled.",
        ],
        "false_dilemma": [
            "You're either with us on {issue}, or you're part of the problem. There's no middle ground.",
            "We have two choices: {option_a} or {option_b}. Anyone who suggests otherwise is deluding themselves.",
            "It's simple — {action_a} or {disaster}. Those are the only options on the table.",
        ],
        "ad_hominem": [
            "Of course {person} would say that — they've been {negative_trait} their entire career. Why would anyone trust a {insult}?",
            "Consider the source: {person} has a history of {negative_history}. Their opinion on {topic} is worthless.",
            "Let's not forget that {person} was once {embarrassing_fact}. And now they want us to believe them about {topic}?",
        ],
        "whataboutism": [
            "Why are we talking about {topic_a} when {entity} has done far worse with {topic_b}? That's the real scandal.",
            "You criticize {target} for {action_a}? What about {other_entity} doing {action_b}? That's the real hypocrisy.",
            "Before pointing fingers at {target}, maybe look at {entity}'s record on {topic}. Glass houses, people.",
        ],
        "causal_oversimplification": [
            "The reason for {problem} is simple: {single_cause}. If we fix that, everything else falls into place.",
            "{Problem} exists because of one thing and one thing only: {cause}. Anyone who says it's more complicated is lying.",
            "It's not rocket science — {cause} led directly to {effect}. Period. End of story.",
        ],
        "straw_man": [
            "So what they're really saying is that {distorted_claim}. That's absurd, and anyone can see why.",
            "My opponent wants you to believe {exaggerated_position}. Is that really what we want for our {group}?",
            "If we follow their logic, we'd end up {extreme_consequence}. That's what they're actually proposing.",
        ],
        "red_herring": [
            "The real question isn't {actual_topic} — it's {diversion}. That's what we should be focusing on.",
            "Everyone's talking about {topic}, but nobody mentions {unrelated_topic}. That's the story the media won't cover.",
            "Sure, {topic} is important, but have you considered {tangent}? That's where the real problem lies.",
        ],
        "appeal_to_emotion": [
            "Think of the {sympathetic_group} who {emotional_scenario}. Can you look them in the eye and say you don't care?",
            "I've seen {heartbreaking_scenario} firsthand. The {suffering} is real, and it's happening because of {cause}.",
            "My {relative} {sad_story}. This isn't about politics — it's about {emotional_value}.",
        ],
        "repetition": [
            "{Claim}. I'll say it again: {claim}. And one more time for those in the back: {claim}.",
            "Make no mistake: {statement}. {statement}. {statement}. The facts speak for themselves.",
            "We need {action}. We want {action}. We demand {action}. Nothing less will do.",
        ],
        "exaggeration": [
            "This is the most {superlative} {event} in the history of {domain}. Nothing even comes close.",
            "Absolutely {extreme_adj} — the {worst/best} we've ever seen. A {historic_adj} moment of {magnitude}.",
            "Not just {moderate_claim} — this is {extreme_claim} on a scale never before witnessed in {domain}.",
        ],
        "doubt": [
            "Can we really trust {target} after everything that's happened? Something doesn't add up about their {claim}.",
            "The so-called {experts} want you to believe {claim}. But who's funding their research? Follow the money.",
            "I'm not saying {conspiracy}, but isn't it suspicious that {coincidence}? You have to wonder...",
        ],
        "slogans": [
            "{Slogan}! That's our rallying cry. Remember: {slogan}. Together we {action}!",
            "It all comes down to this: {catchy_phrase}. Say it with me: {catchy_phrase}!",
            "Our message is clear: {slogan}. From coast to coast: {slogan}. The future starts now.",
        ],
        "name_calling": [
            "These {derogatory_term} have been {negative_action} for too long. It's time to call them what they are: {insult}.",
            "Only a {insult} would support {position}. The {group} pushing this agenda are nothing but {derogatory}.",
            "Let's be honest about {person} — they're a {insult}, a {insult}, and everyone knows it.",
        ],
        "flag_waving": [
            "As true {nationality}, we must {action}! Our {national_symbol} demands it. For {country} and for freedom!",
            "Our {ancestors} didn't fight and die for us to {unpatriotic_action}. Stand up for {country}!",
            "This is about more than politics — it's about our {national_identity}. Every patriot knows that {claim}.",
        ],
        "thought_terminating_cliche": [
            "It is what it is. That's just how the {domain} works. No point overthinking it.",
            "At the end of the day, {simple_conclusion}. That's the bottom line, and we just need to accept it.",
            "You can't fight {entity}. The system is the system. It's better to just {passive_action}.",
        ],
    },
    "fr": {
        "appeal_to_fear": [
            "Si nous n'agissons pas maintenant, {catastrophe} sera inevitable. Nos {group} en paieront le prix pendant des generations.",
            "La menace de {threat} est plus proche que vous ne le pensez. Sans {solution}, c'est {disaster} qui nous attend.",
        ],
        "loaded_language": [
            "Cette politique {negative_adj} n'est rien d'autre qu'un {negative_noun} concu pour {bad_goal} pendant que {group} {suffer}.",
            "Les forces {adjective} derriere ce {noun} ont {adverb} {verb} tout ce que nous cherrissons.",
        ],
        "bandwagon": [
            "Tout le monde sait que {claim}. La grande majorite de {group} a deja {action}. Ne restez pas en arriere.",
        ],
        "false_dilemma": [
            "Vous etes soit avec nous sur {issue}, soit vous faites partie du probleme. Il n'y a pas de juste milieu.",
        ],
    },
    "es": {
        "appeal_to_fear": [
            "Si no actuamos de inmediato, {catastrophe} sera inevitable. Nuestros {group} sufriran las consecuencias.",
            "La amenaza de {threat} esta mas cerca de lo que crees. Sin {solution}, nos enfrentamos a {disaster}.",
        ],
        "loaded_language": [
            "Esta politica {negative_adj} no es mas que un {negative_noun} disenado para {bad_goal}.",
        ],
        "bandwagon": [
            "Todo el mundo sabe que {claim}. La gran mayoria ya ha {action}. No te quedes atras.",
        ],
    },
    "de": {
        "appeal_to_fear": [
            "Wenn wir nicht sofort {action}, wird {catastrophe} unvermeidlich sein. Unsere {group} werden die Konsequenzen tragen.",
        ],
        "loaded_language": [
            "Diese {negative_adj} Politik ist nichts als ein {negative_noun}, um {bad_goal} zu erreichen.",
        ],
    },
}

NEUTRAL_TEMPLATES = {
    "en": [
        "A recent study published in {journal} found that {finding} among {sample_size} participants over a {duration} period.",
        "The {organization} reported that {metric} changed by {percentage}% in {time_period}, according to official data.",
        "Experts from {institution} presented mixed findings on {topic}, noting both {positive} and {limitation}.",
        "The policy was approved with {vote_count} votes in favor and {against_count} against, after {duration} of debate.",
        "According to the {report_name}, {statistic} in {region} showed {trend} compared to the previous {period}.",
        "The {committee} reviewed evidence from {number} sources before issuing their {type} recommendation on {topic}.",
        "Data from the {survey} indicates that {percentage}% of respondents {response}, while {other_pct}% {other_response}.",
        "The {year} {report} found {finding}. Researchers noted several limitations including {limitation}.",
    ],
    "fr": [
        "Une etude recente publiee dans {journal} a revele que {finding} parmi {sample_size} participants.",
        "L'{organization} a signale que {metric} a change de {percentage}% au cours de {time_period}.",
        "Les experts de {institution} ont presente des resultats mitiges sur {topic}.",
    ],
    "es": [
        "Un estudio reciente publicado en {journal} encontro que {finding} entre {sample_size} participantes.",
        "La {organization} informo que {metric} cambio un {percentage}% en {time_period}.",
    ],
    "de": [
        "Eine aktuelle Studie in {journal} ergab, dass {finding} bei {sample_size} Teilnehmern.",
        "Die {organization} berichtete, dass {metric} sich um {percentage}% in {time_period} veranderte.",
    ],
    "ar": [
        "كشفت دراسة حديثة نشرت في {journal} ان {finding} بين {sample_size} مشاركا.",
        "افاد {organization} ان {metric} تغير بنسبة {percentage}% خلال {time_period}.",
    ],
}

# Slot fillers for templates
SLOT_FILLERS = {
    "action": ["reform the system", "pass this legislation", "change course", "invest in infrastructure", "protect our borders"],
    "catastrophe": ["economic collapse", "social breakdown", "environmental disaster", "a public health crisis", "widespread suffering"],
    "group": ["citizens", "families", "workers", "children", "communities", "taxpayers", "veterans"],
    "threat": ["economic recession", "rising crime", "foreign interference", "technological displacement", "social instability"],
    "solution": ["immediate action", "bold reform", "strong leadership", "collective effort", "decisive measures"],
    "disaster": ["total collapse", "unprecedented crisis", "irreversible damage", "catastrophic failure", "mass suffering"],
    "enemy": ["the establishment", "foreign powers", "corporate elites", "radical activists", "bureaucrats"],
    "bad_action": ["undermining our values", "eroding our freedoms", "destroying our economy", "dividing our communities"],
    "good_action": ["protect ourselves", "save our future", "defend our rights", "preserve our way of life"],
    "number": ["thousands", "millions", "hundreds of thousands", "countless"],
    "victims": ["families", "workers", "children", "communities", "citizens"],
    "suffer": ["lose their homes", "go without healthcare", "face unemployment", "live in fear"],
    "negative_adj": ["disastrous", "reckless", "shameful", "corrupt", "dangerous"],
    "negative_noun": ["scam", "power grab", "betrayal", "conspiracy", "assault"],
    "bad_goal": ["enrich the elite", "silence opposition", "control the masses", "destroy competition"],
    "adjective": ["sinister", "powerful", "shadowy", "corrupt", "ruthless"],
    "noun": ["scheme", "agenda", "operation", "campaign", "movement"],
    "adverb": ["systematically", "ruthlessly", "deliberately", "quietly", "aggressively"],
    "verb": ["undermined", "destroyed", "corrupted", "manipulated", "stolen"],
    "loaded_term": ["war", "attack", "crusade", "genocide", "holocaust"],
    "victim": ["the people", "hardworking families", "innocent children", "our veterans", "the middle class"],
    "extreme_noun": ["tyranny", "oppression", "fascism", "betrayal", "atrocity"],
    "value": ["freedom", "democracy", "justice", "our way of life", "human dignity"],
    "claim": ["this policy will transform our economy", "the current system is broken", "change is overdue"],
    "percentage": ["82", "91", "76", "88", "95"],
    "authority": ["Dr. James Wilson, Nobel laureate", "the Harvard Research Institute", "the World Health Organization"],
    "credentials": ["decades of research", "three PhDs", "40 years of experience"],
    "name": ["Harrison", "Nakamura", "Fernandez", "Singh", "Muller"],
    "institution": ["MIT", "Stanford", "Oxford", "the WHO", "the UN"],
    "title": ["neuroscientist", "economist", "epidemiologist", "policy analyst"],
    "experts": ["scientists", "economists", "analysts", "doctors", "researchers"],
    "issue": ["immigration reform", "economic policy", "climate action", "healthcare"],
    "option_a": ["support this bill", "embrace reform", "act now", "make sacrifices"],
    "option_b": ["accept total defeat", "watch everything crumble", "surrender our future"],
    "action_a": ["embrace the change", "stand with us"],
    "person": ["Senator Johnson", "the CEO", "the opposition leader", "the critic"],
    "negative_trait": ["dishonest", "incompetent", "corrupt", "self-serving"],
    "insult": ["fraud", "hypocrite", "puppet", "liar"],
    "negative_history": ["scandals", "failures", "broken promises", "corruption"],
    "topic": ["the economy", "healthcare", "education", "foreign policy"],
    "embarrassing_fact": ["caught lying under oath", "fired for misconduct", "bankrupt twice"],
    "topic_a": ["our policy", "this issue", "our mistakes"],
    "entity": ["the opposition", "other countries", "the previous administration"],
    "topic_b": ["the real corruption", "systematic abuses", "far greater crimes"],
    "target": ["us", "our party", "this administration"],
    "action_a_wh": ["minor infractions", "small mistakes"],
    "other_entity": ["they", "the other side", "the establishment"],
    "action_b": ["far worse", "systematic corruption", "actual crimes"],
    "problem": ["poverty", "crime", "unemployment", "inflation", "inequality"],
    "single_cause": ["bad policy", "immigration", "deregulation", "corporate greed", "government spending"],
    "cause": ["one failed policy", "a single decision", "their incompetence"],
    "effect": ["the crisis we see today", "this disaster", "our current situation"],
    "distorted_claim": ["we should have no rules at all", "nobody should work", "children don't need education"],
    "exaggerated_position": ["total anarchy is the goal", "we should abolish all regulations"],
    "extreme_consequence": ["living in complete chaos", "with no rights whatsoever", "in total poverty"],
    "actual_topic": ["the policy debate", "the real issue", "what matters"],
    "diversion": ["who's really pulling the strings", "what happened last year", "the bigger picture"],
    "unrelated_topic": ["the media bias", "celebrity scandals", "sports controversies"],
    "tangent": ["the influence of big tech", "what other countries are doing", "historical parallels"],
    "sympathetic_group": ["children", "elderly veterans", "struggling families", "orphans"],
    "emotional_scenario": ["go to bed hungry every night", "can't afford their medication", "lost everything"],
    "heartbreaking_scenario": ["families torn apart", "children crying", "people losing hope"],
    "suffering": ["pain", "despair", "hardship", "devastation"],
    "relative": ["grandmother", "father", "sister", "neighbor"],
    "sad_story": ["couldn't afford surgery", "lost their home", "was denied help"],
    "emotional_value": ["human decency", "compassion", "our shared humanity"],
    "statement": ["this must change", "we deserve better", "enough is enough"],
    "superlative": ["catastrophic", "unprecedented", "historic", "revolutionary", "shocking"],
    "event": ["crisis", "scandal", "achievement", "disaster", "breakthrough"],
    "domain": ["modern politics", "our country", "the 21st century", "world history"],
    "extreme_adj": ["devastating", "extraordinary", "unbelievable", "earth-shattering"],
    "historic_adj": ["epoch-defining", "once-in-a-lifetime", "generation-defining"],
    "magnitude": ["biblical proportions", "staggering scale", "unprecedented scope"],
    "moderate_claim": ["a small increase", "a minor issue", "a slight concern"],
    "extreme_claim": ["a total catastrophe", "an absolute disaster", "complete annihilation"],
    "conspiracy": ["it's a cover-up", "they're hiding something", "there's a conspiracy"],
    "coincidence": ["it all happened at the same time", "the timing is perfect", "no one investigated"],
    "Slogan": ["Power to the People", "No More Lies", "Stand and Fight", "Take Back Control"],
    "slogan": ["power to the people", "no more lies", "stand and fight", "take back control"],
    "catchy_phrase": ["Enough is enough", "United we stand", "Change starts now", "Truth over fear"],
    "derogatory_term": ["elitists", "radicals", "cronies", "puppets", "sellouts"],
    "negative_action": ["betraying the public trust", "enriching themselves", "destroying our values"],
    "derogatory": ["parasites", "traitors", "snakes", "thieves"],
    "position": ["this absurd plan", "such nonsense", "this disaster"],
    "nationality": ["Americans", "citizens", "patriots", "sons and daughters of this nation"],
    "national_symbol": ["flag", "constitution", "heritage", "founding principles"],
    "country": ["our great nation", "this country", "the republic", "the homeland"],
    "ancestors": ["founders", "forefathers", "brave soldiers", "heroes of the past"],
    "unpatriotic_action": ["give up our sovereignty", "surrender our values", "abandon our principles"],
    "national_identity": ["national pride", "cultural heritage", "patriotic duty"],
    "simple_conclusion": ["things just work out", "what's done is done", "life goes on"],
    "passive_action": ["go along with it", "accept reality", "move on"],
    # Neutral fillers
    "journal": ["Nature", "The Lancet", "Science", "PNAS", "BMJ"],
    "finding": ["moderate improvement in outcomes", "no statistically significant difference", "a correlation between the variables"],
    "sample_size": ["5,000", "12,000", "850", "23,400", "2,100"],
    "duration": ["6-month", "2-year", "10-year", "5-year"],
    "organization": ["Bureau of Statistics", "Central Bank", "Research Council", "OECD"],
    "metric": ["the unemployment rate", "GDP growth", "average household income", "inflation"],
    "time_period": ["Q3 2025", "the fiscal year", "the past decade"],
    "positive": ["increased efficiency", "improved access", "cost savings"],
    "limitation": ["sample size constraints", "regional variation", "data availability"],
    "vote_count": ["312", "287", "198", "156"],
    "against_count": ["245", "198", "134", "89"],
    "report_name": ["Annual Economic Review", "Global Health Report", "Census Data"],
    "statistic": ["median household income", "life expectancy", "GDP per capita"],
    "region": ["the northern provinces", "urban areas", "OECD countries"],
    "trend": ["a 2.3% increase", "relative stability", "a modest decline"],
    "period": ["quarter", "year", "decade"],
    "committee": ["advisory panel", "review board", "scientific committee"],
    "survey": ["National Household Survey", "Gallup Poll", "Eurobarometer"],
    "response": ["supported the measure", "reported improvement", "agreed"],
    "other_pct": ["34", "22", "41", "28"],
    "other_response": ["expressed concerns", "disagreed", "were uncertain"],
    "year": ["2025", "2024", "2023"],
    "report": ["meta-analysis", "systematic review", "longitudinal study"],
    "type": ["preliminary", "final", "interim", "conditional"],
}


class TemplateGenerator:
    """Generate examples from parameterized templates."""

    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages or ["en"]

    def generate(self, target_count: int) -> list[SyntheticExample]:
        """Generate `target_count` examples using templates."""
        examples: list[SyntheticExample] = []
        persuasive_ratio = 0.55  # 55% persuasive, 45% neutral

        n_persuasive = int(target_count * persuasive_ratio)
        n_neutral = target_count - n_persuasive

        logger.info(f"Template generation: {n_persuasive} persuasive + {n_neutral} neutral")

        # --- Persuasive examples ---
        for _ in tqdm(range(n_persuasive), desc="Persuasive templates"):
            lang = random.choice(self.languages)
            lang_templates = PERSUASIVE_TEMPLATES.get(lang, PERSUASIVE_TEMPLATES["en"])

            # Sample 1-3 techniques with weighted sampling
            available = [t for t in TECHNIQUE_LABELS if t in lang_templates]
            if not available:
                available = list(lang_templates.keys())
            weights = [TECHNIQUE_WEIGHTS.get(t, 1.0) for t in available]
            n_techniques = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
            techniques = random.choices(available, weights=weights, k=min(n_techniques, len(available)))
            techniques = list(set(techniques))

            # Pick a template from one of the techniques
            primary_tech = techniques[0]
            templates = lang_templates.get(primary_tech, [])
            if not templates:
                continue

            template = random.choice(templates)
            text = self._fill_template(template)
            domain = random.choice(DOMAINS)
            tone = random.choice(TONES)

            difficulty = random.choices(["easy", "medium", "hard"], weights=[0.3, 0.5, 0.2])[0]

            examples.append(SyntheticExample(
                text=text,
                language=lang,
                techniques=techniques,
                is_persuasive=True,
                domain=domain,
                tone=tone,
                generation_method="template",
                difficulty=difficulty,
            ))

        # --- Neutral examples ---
        for _ in tqdm(range(n_neutral), desc="Neutral templates"):
            lang = random.choice(self.languages)
            templates = NEUTRAL_TEMPLATES.get(lang, NEUTRAL_TEMPLATES["en"])
            template = random.choice(templates)
            text = self._fill_template(template)

            examples.append(SyntheticExample(
                text=text,
                language=lang,
                techniques=[],
                is_persuasive=False,
                domain=random.choice(DOMAINS),
                generation_method="template",
                difficulty="easy",
            ))

        random.shuffle(examples)
        return examples

    @staticmethod
    def _fill_template(template: str) -> str:
        """Fill a template string with random slot values."""
        import re
        filled = template
        max_iterations = 20
        for _ in range(max_iterations):
            match = re.search(r"\{(\w+)\}", filled)
            if not match:
                break
            slot = match.group(1)
            fillers = SLOT_FILLERS.get(slot, [f"[{slot}]"])
            replacement = random.choice(fillers)
            filled = filled[:match.start()] + replacement + filled[match.end():]
        return filled


# ---------------------------------------------------------------------------
# LLM-powered generation (high-quality, diverse)
# ---------------------------------------------------------------------------

LLM_GENERATION_PROMPT = """\
You are a dataset generator for a persuasion detection NLP research project.

Generate {batch_size} diverse, realistic examples of text that {task_desc}.

Requirements:
- Language: {language} ({lang_name})
- Domain: {domain}
- Difficulty: {difficulty}
{technique_instructions}
- Each example should be 2-5 sentences long
- Make examples realistic — they should sound like real news articles, social media posts, speeches, or opinion pieces
- Vary the style: some formal, some informal, some emotional, some subtle
- Do NOT be repetitive — each example must be significantly different

Return ONLY a JSON array of objects:
[
  {{
    "text": "The example text...",
    "techniques": ["technique_1", "technique_2"],
    "is_persuasive": true,
    "tone": "aggressive|subtle|emotional|formal|casual",
    "spans": [
      {{"start": 0, "end": 15, "technique": "technique_1", "text": "exact span text"}}
    ]
  }},
  ...
]

IMPORTANT: "spans" must contain the exact character offsets and text of manipulative phrases within the "text" field.
"""


class LLMGenerator:
    """Generate high-quality examples using GPT-4o-mini."""

    def __init__(self, model: str = "gpt-4o-mini", batch_size: int = 10) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key.startswith("sk-"):
                from openai import OpenAI
                self._client = OpenAI()
                logger.info("LLM Generator: OpenAI client ready")
        except Exception as e:
            logger.warning(f"OpenAI not available for generation: {e}")

    def generate(
        self,
        target_count: int,
        languages: list[str] | None = None,
    ) -> list[SyntheticExample]:
        """Generate examples using LLM calls."""
        if not self._client:
            logger.warning("No OpenAI client — skipping LLM generation")
            return []

        languages = languages or ["en"]
        examples: list[SyntheticExample] = []
        total_batches = (target_count + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(total_batches), desc="LLM generation"):
            lang = random.choice(languages)
            domain = random.choice(DOMAINS)
            difficulty = random.choices(
                ["easy", "medium", "hard"], weights=[0.2, 0.5, 0.3]
            )[0]

            # Alternate between persuasive and neutral
            if random.random() < 0.55:
                techniques = random.sample(TECHNIQUE_LABELS, k=random.randint(1, 3))
                tech_instructions = (
                    f"- Techniques to demonstrate: {', '.join(techniques)}\n"
                    f"- The text MUST contain these specific persuasion techniques\n"
                    f"- Include span-level annotations marking exactly where each technique appears"
                )
                task_desc = f"uses persuasion/manipulation techniques ({', '.join(techniques)})"
            else:
                tech_instructions = (
                    "- The text must be NEUTRAL and objective\n"
                    "- No persuasion techniques should be present\n"
                    "- spans should be an empty list"
                )
                task_desc = "is neutral, objective, and factual (no persuasion techniques)"
                techniques = []

            prompt = LLM_GENERATION_PROMPT.format(
                batch_size=min(self.batch_size, target_count - len(examples)),
                task_desc=task_desc,
                language=lang,
                lang_name=LANGUAGE_NAMES.get(lang, lang),
                domain=domain,
                difficulty=difficulty,
                technique_instructions=tech_instructions,
            )

            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content.strip()
                content = content.removeprefix("```json").removesuffix("```").strip()
                items = json.loads(content)

                for item in items:
                    examples.append(SyntheticExample(
                        text=item["text"],
                        language=lang,
                        techniques=item.get("techniques", []),
                        is_persuasive=item.get("is_persuasive", bool(item.get("techniques"))),
                        domain=domain,
                        tone=item.get("tone", ""),
                        generation_method="llm",
                        difficulty=difficulty,
                        spans=item.get("spans", []),
                    ))

            except Exception as e:
                logger.warning(f"LLM batch {batch_idx} failed: {e}")
                time.sleep(2)
                continue

            if len(examples) >= target_count:
                break

            # Rate limiting
            time.sleep(0.5)

        return examples[:target_count]


# ---------------------------------------------------------------------------
# Paraphrase augmentation
# ---------------------------------------------------------------------------

PARAPHRASE_PROMPT = """\
Paraphrase the following text while preserving ALL persuasion techniques present.
The paraphrase should sound natural and different from the original but use the same manipulation strategies.

Original text ({language}):
{text}

Techniques present: {techniques}

Return ONLY a JSON object:
{{
  "paraphrase": "The paraphrased text...",
  "spans": [
    {{"start": 0, "end": 15, "technique": "technique_1", "text": "exact span text"}}
  ]
}}
"""


class ParaphraseAugmenter:
    """Create paraphrased variants of existing examples."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = None
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and api_key.startswith("sk-"):
                from openai import OpenAI
                self._client = OpenAI()
        except Exception:
            pass

    def augment(
        self,
        examples: list[SyntheticExample],
        augment_factor: float = 1.5,
    ) -> list[SyntheticExample]:
        """Create paraphrased versions of persuasive examples."""
        if not self._client:
            logger.warning("No OpenAI client — skipping paraphrase augmentation")
            return []

        persuasive = [e for e in examples if e.is_persuasive]
        n_to_augment = int(len(persuasive) * (augment_factor - 1))
        selected = random.sample(persuasive, min(n_to_augment, len(persuasive)))

        augmented: list[SyntheticExample] = []
        for ex in tqdm(selected, desc="Paraphrasing"):
            try:
                prompt = PARAPHRASE_PROMPT.format(
                    language=LANGUAGE_NAMES.get(ex.language, ex.language),
                    text=ex.text,
                    techniques=", ".join(ex.techniques),
                )
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content.strip()
                content = content.removeprefix("```json").removesuffix("```").strip()
                data = json.loads(content)

                augmented.append(SyntheticExample(
                    text=data["paraphrase"],
                    language=ex.language,
                    techniques=ex.techniques,
                    is_persuasive=True,
                    domain=ex.domain,
                    tone=ex.tone,
                    generation_method="paraphrase",
                    difficulty=ex.difficulty,
                    spans=data.get("spans", []),
                ))
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"Paraphrase failed: {e}")
                continue

        return augmented


# ---------------------------------------------------------------------------
# Dataset assembler
# ---------------------------------------------------------------------------

class DatasetAssembler:
    """Assemble, deduplicate, and export the final dataset."""

    @staticmethod
    def deduplicate(examples: list[SyntheticExample], min_similarity: float = 0.9) -> list[SyntheticExample]:
        """Remove near-duplicate examples using text hashing."""
        seen_hashes: set[str] = set()
        unique: list[SyntheticExample] = []

        for ex in examples:
            # Normalize and hash
            normalized = ex.text.lower().strip()
            h = hashlib.md5(normalized.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(ex)

        removed = len(examples) - len(unique)
        if removed:
            logger.info(f"Deduplication: removed {removed} duplicates ({len(unique)} remaining)")
        return unique

    @staticmethod
    def validate(examples: list[SyntheticExample]) -> list[SyntheticExample]:
        """Validate and filter examples."""
        valid: list[SyntheticExample] = []
        for ex in examples:
            if not ex.text or len(ex.text.strip()) < 20:
                continue
            if ex.is_persuasive and not ex.techniques:
                continue
            if not ex.is_persuasive and ex.techniques:
                ex.techniques = []
            # Validate techniques are from the catalog
            ex.techniques = [t for t in ex.techniques if t in TECHNIQUE_LABELS]
            valid.append(ex)

        removed = len(examples) - len(valid)
        if removed:
            logger.info(f"Validation: removed {removed} invalid examples ({len(valid)} remaining)")
        return valid

    @staticmethod
    def compute_statistics(examples: list[SyntheticExample]) -> dict:
        """Compute dataset statistics."""
        total = len(examples)
        persuasive = sum(1 for e in examples if e.is_persuasive)
        neutral = total - persuasive

        lang_dist = {}
        tech_dist = {}
        method_dist = {}
        domain_dist = {}
        difficulty_dist = {}
        span_count = 0

        for ex in examples:
            lang_dist[ex.language] = lang_dist.get(ex.language, 0) + 1
            method_dist[ex.generation_method] = method_dist.get(ex.generation_method, 0) + 1
            if ex.domain:
                domain_dist[ex.domain] = domain_dist.get(ex.domain, 0) + 1
            difficulty_dist[ex.difficulty] = difficulty_dist.get(ex.difficulty, 0) + 1
            span_count += len(ex.spans)
            for t in ex.techniques:
                tech_dist[t] = tech_dist.get(t, 0) + 1

        return {
            "total_examples": total,
            "persuasive": persuasive,
            "neutral": neutral,
            "persuasive_ratio": round(persuasive / max(total, 1), 3),
            "languages": lang_dist,
            "techniques": dict(sorted(tech_dist.items(), key=lambda x: -x[1])),
            "generation_methods": method_dist,
            "domains": domain_dist,
            "difficulty": difficulty_dist,
            "total_span_annotations": span_count,
            "avg_techniques_per_persuasive": round(
                sum(len(e.techniques) for e in examples if e.is_persuasive) / max(persuasive, 1), 2
            ),
        }

    @staticmethod
    def export(
        examples: list[SyntheticExample],
        output_dir: Path,
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> dict[str, Path]:
        """Export dataset with train/val/test splits."""
        output_dir.mkdir(parents=True, exist_ok=True)

        random.shuffle(examples)
        n = len(examples)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        splits = {
            "train": examples[:n_train],
            "val": examples[n_train:n_train + n_val],
            "test": examples[n_train + n_val:],
        }

        paths = {}
        for split_name, split_data in splits.items():
            path = output_dir / f"{split_name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for ex in split_data:
                    f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            paths[split_name] = path
            logger.info(f"  {split_name}: {len(split_data)} examples -> {path}")

        # Also export full dataset as DataFrame
        df_path = output_dir / "full_dataset.parquet"
        df = pd.DataFrame([ex.to_dict() for ex in examples])
        df.to_parquet(df_path, index=False)
        paths["full"] = df_path

        # Export statistics
        stats = DatasetAssembler.compute_statistics(examples)
        stats_path = output_dir / "dataset_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        paths["stats"] = stats_path

        return paths


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic persuasion dataset")
    parser.add_argument("--target-size", type=int, default=200000, help="Target dataset size")
    parser.add_argument("--output", type=str, default="data/synthetic/", help="Output directory")
    parser.add_argument("--languages", type=str, default="en,fr,ar,es,de,zh,hi", help="Comma-separated language codes")
    parser.add_argument("--mode", choices=["full", "templates-only", "llm-only"], default="full")
    parser.add_argument("--llm-ratio", type=float, default=0.3, help="Fraction to generate via LLM")
    parser.add_argument("--augment-factor", type=float, default=1.5, help="Paraphrase augmentation factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    languages = [l.strip() for l in args.languages.split(",")]

    logger.info(f"Target: {args.target_size:,} examples | Languages: {languages} | Mode: {args.mode}")

    all_examples: list[SyntheticExample] = []

    # --- Phase 1: Template generation ---
    if args.mode in ("full", "templates-only"):
        template_count = args.target_size if args.mode == "templates-only" else int(args.target_size * (1 - args.llm_ratio))
        gen = TemplateGenerator(languages=languages)
        template_examples = gen.generate(template_count)
        all_examples.extend(template_examples)
        logger.info(f"Phase 1 (templates): {len(template_examples):,} examples")

    # --- Phase 2: LLM generation ---
    if args.mode in ("full", "llm-only"):
        llm_count = args.target_size if args.mode == "llm-only" else int(args.target_size * args.llm_ratio)
        llm_gen = LLMGenerator(batch_size=10)
        llm_examples = llm_gen.generate(llm_count, languages=languages)
        all_examples.extend(llm_examples)
        logger.info(f"Phase 2 (LLM): {len(llm_examples):,} examples")

    # --- Phase 3: Paraphrase augmentation ---
    if args.mode == "full" and args.augment_factor > 1.0:
        augmenter = ParaphraseAugmenter()
        paraphrased = augmenter.augment(all_examples, args.augment_factor)
        all_examples.extend(paraphrased)
        logger.info(f"Phase 3 (paraphrase): {len(paraphrased):,} additional examples")

    # --- Phase 4: Validate, deduplicate, export ---
    assembler = DatasetAssembler()
    all_examples = assembler.validate(all_examples)
    all_examples = assembler.deduplicate(all_examples)

    stats = assembler.compute_statistics(all_examples)
    logger.info(f"\nFinal dataset: {stats['total_examples']:,} examples")
    logger.info(f"  Persuasive: {stats['persuasive']:,} ({stats['persuasive_ratio']:.1%})")
    logger.info(f"  Neutral: {stats['neutral']:,}")
    logger.info(f"  Languages: {stats['languages']}")
    logger.info(f"  Span annotations: {stats['total_span_annotations']:,}")

    paths = assembler.export(all_examples, output_dir)
    logger.success(f"\nDataset exported to {output_dir}/")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")


if __name__ == "__main__":
    main()
