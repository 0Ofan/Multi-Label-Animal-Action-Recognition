"""Optional LLM semantic search over the fixed action vocabulary.

Search only re-orders and filters the Results view. It never changes scores,
thresholds or emitted labels. Without credentials a bilingual alias/token
matcher keeps the dashboard working offline.
"""
from __future__ import annotations
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_MATCHES = 12
MAX_QUERY_CHARS = 300
_TOKEN = re.compile(r"[a-z0-9]+")

SYSTEM_PROMPT = (
    "You map a free-text search query (English or Chinese) to a fixed animal-action taxonomy. "
    "Use ONLY class IDs from the provided list. Return strict JSON and nothing else: "
    '{"matches":[{"id":<integer>,"reason":"<short reason>"}]}. '
    f"Return at most {MAX_MATCHES} matches, most relevant first. "
    'If nothing is relevant return {"matches":[]}.'
)

# Aliases used only by the offline fallback; the LLM ignores this table.
ALIASES = {
    # English stems
    "climb": ("Climbing", "Abseiling"),
    "abseil": ("Abseiling", "Climbing"),
    "jump": ("Jumping", "Hopping", "Landing"),
    "hop": ("Hopping", "Jumping"),
    "run": ("Running", "Chasing", "Fleeing", "Walking"),
    "chase": ("Chasing", "Running", "Preying"),
    "flee": ("Fleeing", "Escaping", "Retreating"),
    "escap": ("Escaping", "Fleeing", "Retreating"),
    "attack": ("Attacking", "Biting", "Fighting", "Retaliating"),
    "aggress": ("Attacking", "Biting", "Fighting", "Retaliating", "Competing For Dominance"),
    "fight": ("Fighting", "Attacking", "Retaliating", "Competing For Dominance"),
    "bite": ("Biting", "Attacking"),
    "eat": ("Eating", "Being Eaten", "Pecking", "Preying", "Sharing Food"),
    "feed": ("Eating", "Sharing Food", "Preying", "Pecking"),
    "drink": ("Drinking",),
    "sleep": ("Sleeping", "Sleeping in its nest", "Resting"),
    "rest": ("Resting", "Sleeping", "Lying Down", "Keeping still"),
    "fly": ("Flying", "Gliding", "Flapping", "Spreading Wings"),
    "swim": ("Swimming", "Swimming In Circles", "Diving", "Surfacing", "Sinking", "Drifting"),
    "dive": ("Diving", "Sinking", "Surfacing"),
    "sound": ("Barking", "Calling", "Chirping", "Hissing", "Rattling"),
    "bark": ("Barking", "Calling"),
    "sing": ("Chirping", "Calling"),
    "mate": ("Performing Sexual Display", "Performing copulatory mounting", "Unmounting"),
    "courtship": ("Performing Sexual Display", "Performing copulatory mounting", "Unmounting"),
    "carry": ("Carrying", "Being Carried", "Carrying In Mouth", "Being Carried In Mouth"),
    "hunt": ("Preying", "Wrapping Prey", "Wrapping Itself Around Prey", "Biting", "Stinging"),
    "prey": ("Preying", "Wrapping Prey", "Wrapping Itself Around Prey"),
    "groom": ("Grooming", "Preening", "Washing", "Licking", "Rubbing its head"),
    "clean": ("Grooming", "Preening", "Washing", "Licking"),
    "nest": ("Building nest", "Entering its nest", "Exiting Nest", "Sleeping in its nest"),
    "play": ("Playing", "Playing Dead", "Doing somersault", "Rolling"),
    "dead": ("Dead", "Dying", "Playing Dead", "Immobilized"),
    "die": ("Dying", "Dead"),
    "walk": ("Walking", "Moving", "Turning Around"),
    "sit": ("Sitting", "Squatting", "Lying Down"),
    "stand": ("Standing", "Standing in alert", "Leaning"),
    "egg": ("Laying Eggs", "Exiting Cocoon"),
    "birth": ("Giving Birth", "Laying Eggs"),
    "danger": ("Standing in alert", "Displaying Defensive Pose", "Defensive Rearing", "Startled"),
    "defend": ("Displaying Defensive Pose", "Defensive Rearing", "Retaliating", "Standing in alert"),
    # Chinese
    "攀爬": ("Climbing", "Abseiling"),
    "爬": ("Climbing", "Abseiling"),
    "跳跃": ("Jumping", "Hopping", "Landing"),
    "跳": ("Jumping", "Hopping", "Landing"),
    "奔跑": ("Running", "Chasing", "Fleeing"),
    "跑": ("Running", "Chasing", "Fleeing"),
    "追": ("Chasing", "Running"),
    "逃": ("Escaping", "Fleeing", "Retreating"),
    "攻击": ("Attacking", "Biting", "Fighting", "Retaliating"),
    "打斗": ("Fighting", "Attacking", "Retaliating"),
    "咬": ("Biting", "Attacking"),
    "吃": ("Eating", "Being Eaten", "Pecking", "Preying"),
    "进食": ("Eating", "Pecking", "Preying"),
    "喝": ("Drinking",),
    "睡": ("Sleeping", "Sleeping in its nest", "Resting"),
    "休息": ("Resting", "Sleeping", "Lying Down", "Keeping still"),
    "飞": ("Flying", "Gliding", "Flapping", "Spreading Wings"),
    "游泳": ("Swimming", "Swimming In Circles", "Diving", "Surfacing", "Sinking"),
    "游": ("Swimming", "Swimming In Circles", "Diving", "Surfacing"),
    "叫": ("Barking", "Calling", "Chirping", "Hissing", "Rattling"),
    "吠": ("Barking", "Calling"),
    "鸣": ("Chirping", "Calling", "Rattling"),
    "求偶": ("Performing Sexual Display",),
    "交配": ("Performing copulatory mounting", "Unmounting"),
    "携带": ("Carrying", "Being Carried", "Carrying In Mouth", "Being Carried In Mouth"),
    "抱": ("Hugging", "Holding Hands", "Carrying"),
    "梳毛": ("Grooming", "Preening"),
    "清洁": ("Grooming", "Preening", "Washing"),
    "筑巢": ("Building nest",),
    "巢": ("Building nest", "Entering its nest", "Exiting Nest", "Sleeping in its nest"),
    "玩耍": ("Playing", "Playing Dead", "Doing somersault", "Rolling"),
    "捕食": ("Preying", "Wrapping Prey", "Wrapping Itself Around Prey", "Biting", "Stinging"),
    "猎物": ("Preying", "Wrapping Prey", "Wrapping Itself Around Prey"),
    "产卵": ("Laying Eggs",),
    "分娩": ("Giving Birth",),
    "走": ("Walking", "Moving", "Turning Around"),
    "坐": ("Sitting", "Squatting", "Lying Down"),
    "站": ("Standing", "Standing in alert", "Leaning"),
    "躺": ("Lying Down", "Resting"),
    "死": ("Dead", "Dying", "Playing Dead"),
    "警告": ("Standing in alert", "Displaying Defensive Pose", "Rattling"),
    "威吓": ("Displaying Defensive Pose", "Defensive Rearing", "Puffing its throat", "Standing in alert"),
    "移动": ("Moving", "Walking", "Running", "Turning Around"),
}


def _first(secrets, *paths):
    for path in paths:
        value = secrets
        try:
            for key in path:
                value = value[key]
        except Exception:
            continue
        if value not in (None, ""):
            return value
    return None


def llm_settings(secrets=None):
    """Return OpenAI-compatible settings, or None when no API key is configured."""
    api_key = os.environ.get("ANIMALACT2_LLM_API_KEY") or _first(secrets, ("llm", "api_key"), ("llm_api_key",), ("api_key",))
    if not api_key:
        return None
    base_url = os.environ.get("ANIMALACT2_LLM_BASE_URL") or _first(secrets, ("llm", "base_url"), ("llm_base_url",)) or DEFAULT_BASE_URL
    model = os.environ.get("ANIMALACT2_LLM_MODEL") or _first(secrets, ("llm", "model"), ("llm_model",)) or DEFAULT_MODEL
    return {"api_key": str(api_key).strip(), "base_url": str(base_url).strip().rstrip("/"), "model": str(model).strip()}


def _pairs(vocabulary):
    for item in vocabulary:
        yield (item["class_id"], item["name"]) if isinstance(item, dict) else tuple(item)


def _stem_match(token, name_tokens):
    for candidate in name_tokens:
        if token == candidate:
            return True
        if len(token) >= 3 and candidate.startswith(token):
            return True
        if len(candidate) >= 3 and token.startswith(candidate):
            return True
    return False


def local_search(query, vocabulary):
    """Bilingual token/alias fallback; returns (ranked_ids, reasons)."""
    text = query.casefold()
    tokens = {token for token in _TOKEN.findall(text) if len(token) >= 3}
    names = dict(_pairs(vocabulary))
    by_name = {name: cid for cid, name in names.items()}
    scores, reasons = {}, {}
    for cid, name in names.items():
        name_tokens = set(_TOKEN.findall(name.casefold()))
        hits = sorted(token for token in tokens if _stem_match(token, name_tokens))
        score = 2.0 * len(hits)
        if text and text in name.casefold():
            score += 3.0
        if score:
            scores[cid] = score
            reasons[cid] = ", ".join(hits) or "name substring"
    for term, targets in ALIASES.items():
        if term in text:
            # Earlier targets are the dominant translation; longer terms are more specific.
            weight = 2.0 + len(term)
            for index, target in enumerate(targets):
                cid = by_name.get(target)
                if cid is None:
                    continue
                scores[cid] = scores.get(cid, 0.0) + weight - 0.1 * index
                reasons[cid] = f"{reasons[cid]}; alias:{term}" if reasons.get(cid) else f"alias:{term}"
    ranked = sorted(scores, key=lambda cid: (-scores[cid], names[cid]))[:MAX_MATCHES]
    return ranked, {cid: reasons[cid] for cid in ranked}


def _json_payload(content):
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM did not return JSON") from None
        return json.loads(text[start:end + 1])


def parse_matches(content, names):
    """Validate an LLM payload against {class_id: name}; drop unknown classes."""
    payload = _json_payload(content)
    items = payload.get("matches", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("LLM response has no matches list")
    by_name = {name.casefold(): cid for cid, name in names.items()}
    ids, reasons = [], {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = None
        try:
            value = int(item.get("id"))
            if value in names:
                cid = value
        except (TypeError, ValueError):
            pass
        if cid is None and item.get("name"):
            cid = by_name.get(str(item["name"]).strip().casefold())
        if cid is None or cid in reasons:
            continue
        ids.append(cid)
        reasons[cid] = str(item.get("reason") or "").strip()[:160]
        if len(ids) >= MAX_MATCHES:
            break
    return ids, reasons


def _endpoint(base_url):
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"


def _chat(settings, messages, timeout):
    payload = json.dumps({"model": settings["model"], "temperature": 0, "max_tokens": 600, "messages": messages}).encode("utf-8")
    request = Request(_endpoint(settings["base_url"]), data=payload,
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + settings["api_key"]})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"network error: {exc}") from exc
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected API response") from exc


def llm_search(query, vocabulary, settings, timeout=20):
    names = dict(_pairs(vocabulary))
    taxonomy = "; ".join(f"{cid}={name}" for cid, name in names.items())
    content = _chat(settings, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\n\nTaxonomy: {taxonomy}"},
    ], timeout)
    return parse_matches(content, names)


def search_actions(query, vocabulary, settings=None, timeout=20):
    """Return provider, ranked class IDs and short reasons. Never raises for API failures."""
    query = (query or "").strip()[:MAX_QUERY_CHARS]
    if not query:
        return {"provider": "none", "ids": [], "reasons": {}, "model": None, "error": None}
    if settings and settings.get("api_key"):
        try:
            ids, reasons = llm_search(query, vocabulary, settings, timeout)
            return {"provider": "llm", "ids": ids, "reasons": reasons, "model": settings.get("model"), "error": None}
        except Exception as exc:
            ids, reasons = local_search(query, vocabulary)
            return {"provider": "local", "ids": ids, "reasons": reasons, "model": None, "error": str(exc)[:200]}
    ids, reasons = local_search(query, vocabulary)
    return {"provider": "local", "ids": ids, "reasons": reasons, "model": None, "error": None}
