"""Offline stand-in for the LLM client: a valid answer for every prompt type, deterministic in (seed, prompt).

It only has to make the whole game, the diary, the survey and the report run without a model, so the answers are
plausible-looking noise, not play. Each reply seeds its own random.Random from the prompt text so the stub never
touches the global random stream that drives the deck and the bullets.
"""
import json
import random
import re

import rosenhan_prompts as P


class StubClient:
    def __init__(self, seed=0):
        self.seed = seed

    def chat(self, messages, model="", max_new_tokens=0):
        text = messages[-1]["content"]
        return self.reply(text, random.Random(f"{self.seed}|{text}")), ""

    @staticmethod
    def reply(text, rng):
        # Our own prompts first: they embed a transcript whose wording the stub does not control.
        others = re.findall(r"Other players: (.*)", text)
        names = [n.strip() for n in others[0].split(",")] if others else []
        if P.DIARY_ASK in text:
            return json.dumps({"diary": f"Stub note {rng.randint(0, 99)}: play looks ordinary."})
        if P.SURVEY_ASK in text:
            return json.dumps({n: dict(p_arrangement=rng.randint(0, 100), p_auditor=rng.randint(0, 100),
                                       reason="stub reason") for n in names})
        if P.REPORT_ASK in text:
            return json.dumps({"colluders": names[:1], "observed": "stub report"})
        if '"played_cards"' in text:
            hand = re.search(r"Your current hand is: (.*)", text).group(1).split(", ")
            target = re.search(r"target card: (\w+)", text).group(1)
            honest = [c for c in hand if c in (target, "Joker")]
            pool = honest if honest and rng.random() < 0.6 else hand
            cards = rng.sample(pool, min(len(pool), rng.choice([1, 2])))
            return json.dumps({"played_cards": cards, "behavior": "stub calm", "play_reason": "stub reason"})
        if '"was_challenged"' in text:
            return json.dumps({"was_challenged": rng.random() < 0.4, "challenge_reason": "stub reason"})
        if "Please update your overall impression" in text:
            return "Stub impression: plays unremarkably."
        if "SPECIAL OPPORTUNITY" in text:
            return "ACCEPT - PARTNER: Luke - stub" if rng.random() < 0.7 else "REFUSE - stub"
        if "SECRET MESSAGE FROM" in text:
            return "Understood, stub reply."
        if "SECRET CHANNEL COMMUNICATION" in text:
            return "Stub: I hold two target cards, do not challenge me."
        raise ValueError(f"stub got an unknown prompt type: {text[:120]!r}")
