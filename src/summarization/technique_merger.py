"""Technique-level deduplication: merge near-duplicate canonical techniques.

A "technique" here is the canonical ``technique_name`` a group of tricks
shares (see ``build_deterministic_mapping`` in ``normalizer.py``). Near-
duplicate techniques — different spellings, synonyms, or chatty variants of
the same attack — should render as a single output file so that trick-level
dedup and retrieval operate on clean groups.

Merges are applied in three tiers, each verified before being committed:

1. **Content evidence** — two groups share >= ``CONTENT_SHARED_TITLES``
   identical trick-title token sets (e.g. ``type_juggling`` /
   ``php_loose_comparison``, the ``jwt`` family). Shared *key_code* is NOT a
   merge trigger: common payloads (``cat /etc/passwd``) collide across
   genuinely distinct techniques (``sql_injection`` vs ``path_traversal``).
2. **Name similarity** — fuzzy ratio >= ``AUTO_MERGE_RATIO`` AND high token
   overlap (word-Dice >= ``AUTO_MERGE_DICE`` or token containment). Catches
   inflection/split variants like ``php_disable_functions_bypass`` /
   ``php_disabled_functions_bypass`` while NOT collapsing ``aslr_bypass`` /
   ``kaslr_bypass`` (Dice 0.5, no containment).
3. **LLM judge** (opt-in, async) — ambiguous name-similar pairs
   (ratio 85-90 with some title-token overlap) where name similarity alone
   cannot decide. The system prompt is static and cacheable; pair data goes in
   the user message (project prompt-caching convention).

Everything else stays separate. Shared source URLs are treated as noise for
technique merging: one writeup legitimately covers many distinct attacks
(e.g. ``path_traversal`` and ``sql_injection`` share ~1700 sources).
"""

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar

from rapidfuzz import fuzz

from src.config import settings
from src.models import Trick
from src.summarization.normalizer import DEFAULT_MAPPING

logger = logging.getLogger(__name__)


@dataclass
class _Signature:
    """Content summary of one canonical technique group."""

    count: int = 0
    #: multiset of trick-title token sets (identical title -> multiplicity)
    titles: Counter = field(default_factory=Counter)
    #: bag of all title words (for title-token Jaccard)
    title_tokens: Counter = field(default_factory=Counter)
    #: key_code token sets (supporting evidence only, never a merge trigger)
    codes: set = field(default_factory=set)
    #: URLs of the source writeups (noise signal for merging)
    urls: set = field(default_factory=set)


@dataclass
class TechniqueMergeResult:
    """Outcome of a technique-merge pass."""

    #: old_canonical -> merged_canonical (identity for unmerged)
    merged_mapping: dict[str, str] = field(default_factory=dict)
    #: each component's member canonicals, representative first
    components: list[list[str]] = field(default_factory=list)
    #: pairs the LLM decided to KEEP separate (for the user to review)
    llm_kept: list[tuple[str, str]] = field(default_factory=list)
    #: pair counts per decision tier
    stats: dict[str, int] = field(default_factory=dict)
    #: name-similar pairs pending an LLM decision (empty after judge runs)
    llm_candidates: list[tuple[str, str]] = field(default_factory=list)


def _token_set(text: str) -> frozenset:
    """Word token-set of a string (lowercased)."""
    return frozenset(re.findall(r"\w+", (text or "").lower()))


def name_similarity(a: str, b: str) -> tuple[float, float]:
    """Return (word-Dice, fuzzy ratio) between two canonical names."""
    wa, wb = set(a.split("_")), set(b.split("_"))
    if not wa or not wb:
        dice = 0.0
    else:
        dice = 2 * len(wa & wb) / (len(wa) + len(wb))
    return dice, fuzz.ratio(a, b)


def _title_tokens_jaccard(sa: _Signature, sb: _Signature) -> float:
    """Multiset Jaccard over title word bags (shared / union)."""
    a, b = sa.title_tokens, sb.title_tokens
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


class TechniqueMerger:
    """Merge near-duplicate canonical techniques into a deterministic mapping."""

    NAME_CANDIDATE_RATIO = 85
    AUTO_MERGE_RATIO = 90
    AUTO_MERGE_DICE = 0.7
    CONTENT_SHARED_TITLES = 2
    LLM_CONTENT_JACCARD = 0.1
    LLM_BATCH_SIZE = 20

    #: one-token swaps that are genuinely different techniques (e.g. overflow
    #: vs underflow) — never auto-merge on name similarity alone
    ANTIPODAL_TOKENS: ClassVar[dict[str, str]] = {
        "overflow": "underflow",
        "encode": "decode",
        "encrypt": "decrypt",
        "obfuscate": "deobfuscate",
        "obfuscation": "deobfuscation",
        "compress": "decompress",
        "pack": "unpack",
        "zip": "unzip",
        "serialize": "deserialize",
        "serialization": "deserialization",
        "remote": "local",
        "read": "write",
        "patch": "unpatch",
        "sign": "verify",
    }

    def __init__(
        self,
        name_threshold: int | None = None,
        content_shared_titles: int | None = None,
        llm_content_jaccard: float | None = None,
        canonical_values: Iterable[str] | None = None,
        use_llm: bool = True,
        llm_client=None,
    ):
        self.name_threshold = (
            name_threshold or settings.technique_name_sim_threshold or self.NAME_CANDIDATE_RATIO
        )
        self.content_shared = (
            content_shared_titles
            or settings.technique_content_shared_titles
            or self.CONTENT_SHARED_TITLES
        )
        self.llm_content_jaccard = (
            llm_content_jaccard
            or settings.technique_llm_content_jaccard
            or self.LLM_CONTENT_JACCARD
        )
        #: names we prefer as merge representatives (DEFAULT_MAPPING canonicals)
        self.canonical_values = set(canonical_values or DEFAULT_MAPPING.values())
        self.use_llm = use_llm
        self.llm_client = llm_client
        #: pairs the LLM judged as KEEP (set during judge_pairs)
        self._llm_kept: list[tuple[str, str]] = []
        #: signatures from the last build_merged_mapping call
        self._sigs: dict[str, _Signature] = {}

    @property
    def llm_kept(self) -> list[tuple[str, str]]:
        """Pairs the LLM decided to keep separate (for review)."""
        return list(self._llm_kept)

    # -- signatures ---------------------------------------------------------

    def build_signatures(self, groups: dict[str, list[Trick]]) -> dict[str, _Signature]:
        """Summarize each canonical group: titles, codes, urls, count."""
        sigs: dict[str, _Signature] = {}
        for canonical, tricks in groups.items():
            sig = _Signature(count=len(tricks))
            for trick in tricks:
                sig.titles[_token_set(trick.title)] += 1
                sig.title_tokens.update(re.findall(r"\w+", trick.title.lower()))
                if trick.key_code:
                    sig.codes.add(_token_set(trick.key_code))
                sig.urls.update(trick.source_writeups)
            sigs[canonical] = sig
        return sigs

    # -- candidate generation ------------------------------------------------

    def find_name_pairs(self, sigs: dict[str, _Signature]) -> set[tuple[str, str]]:
        """Pairs sharing >=1 name token with fuzzy ratio >= threshold.

        Token blocking (rather than all-pairs) keeps this cheap: two names
        must share a word to be compared, which bounds the candidate space.
        """
        token_index: dict[str, set[str]] = defaultdict(set)
        for name in sigs:
            for token in set(name.split("_")):
                token_index[token].add(name)

        pairs: set[tuple[str, str]] = set()
        for name in sigs:
            for token in set(name.split("_")):
                for other in token_index[token]:
                    if other == name:
                        continue
                    a, b = (name, other) if name < other else (other, name)
                    if (a, b) in pairs:
                        continue
                    if fuzz.ratio(a, b) >= self.name_threshold:
                        pairs.add((a, b))
        return pairs

    def find_content_pairs(self, sigs: dict[str, _Signature]) -> dict[tuple[str, str], int]:
        """Pairs sharing >= CONTENT_SHARED_TITLES identical title token sets.

        Returns {pair: shared_title_count}. Code overlap is intentionally not
        a trigger (common payloads collide across distinct techniques).
        """
        title_index: dict[frozenset, set[str]] = defaultdict(set)
        for name, sig in sigs.items():
            for title in sig.titles:
                title_index[title].add(name)

        pairs: dict[tuple[str, str], int] = {}
        for title, names in title_index.items():
            names = sorted(names)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    pair = (a, b)
                    # distinct shared titles: each bucket contributes one
                    pairs[pair] = pairs.get(pair, 0) + 1
        return {p: n for p, n in pairs.items() if n >= self.content_shared}

    # -- classification ------------------------------------------------------

    def classify(
        self,
        sigs: dict[str, _Signature],
        name_pairs: set[tuple[str, str]],
        content_pairs: dict[tuple[str, str], int],
    ) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
        """Decide each candidate pair -> (auto_merge, llm, keep).

        Content evidence always wins. Name-similar pairs auto-merge only when
        near-identical (ratio/dice/containment); ambiguous ones are deferred to
        the LLM judge; the rest are kept separate.
        """
        auto: set[tuple[str, str]] = set()
        llm: set[tuple[str, str]] = set()
        keep: set[tuple[str, str]] = set()

        # Content candidates are merged outright (already distinct-title filtered).
        auto.update(content_pairs)

        for a, b in name_pairs:
            if (a, b) in content_pairs:
                continue  # already handled
            dice, ratio = name_similarity(a, b)
            wa, wb = set(a.split("_")), set(b.split("_"))
            contained = wa <= wb or wb <= wa
            antipodal = self._is_antipodal_swap(a, b)
            if (
                not antipodal
                and ratio >= self.AUTO_MERGE_RATIO
                and (dice >= self.AUTO_MERGE_DICE or contained)
            ):
                auto.add((a, b))
                continue
            if ratio >= self.name_threshold and not self.use_llm:
                # LLM disabled: keep ambiguous name-similar pairs separate.
                keep.add((a, b))
                continue
            if ratio >= self.name_threshold and self.use_llm:
                if _title_tokens_jaccard(sigs[a], sigs[b]) >= self.llm_content_jaccard:
                    llm.add((a, b))
                    continue
                keep.add((a, b))
                continue
            keep.add((a, b))
        return auto, llm, keep

    # -- merge mapping ---------------------------------------------------------

    def build_merged_mapping(self, groups: dict[str, list[Trick]]) -> TechniqueMergeResult:
        """Compute the merged mapping WITHOUT running the LLM judge.

        Auto/content merges are applied; name-similar pairs are recorded in
        ``result.llm_candidates`` for :meth:`judge_pairs` to decide.
        """
        sigs = self.build_signatures(groups)
        self._sigs = sigs
        name_pairs = self.find_name_pairs(sigs)
        content_pairs = self.find_content_pairs(sigs)
        auto, llm, keep = self.classify(sigs, name_pairs, content_pairs)

        # Union-find over accepted merge pairs -> transitive, deterministic.
        parent = {n: n for n in sigs}

        def find(x: str) -> str:
            root = x
            while parent[root] != root:
                parent[root] = parent[parent[root]]
                root = parent[root]
            return root

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                # deterministic root (lex smaller); representative chosen later
                if rb < ra:
                    ra, rb = rb, ra
                parent[rb] = ra

        for a, b in auto:
            union(a, b)

        grouped: dict[str, list[str]] = defaultdict(list)
        for name in sigs:
            grouped[find(name)].append(name)

        # Choose the representative for each component.
        merged: dict[str, str] = {}
        component_list: list[list[str]] = []
        for members in grouped.values():
            members = sorted(members)
            rep = self._pick_representative(members, sigs)
            component_list.append([rep] + [m for m in members if m != rep])
            for m in members:
                merged[m] = rep

        return TechniqueMergeResult(
            merged_mapping=merged,
            components=component_list,
            llm_candidates=sorted(llm),
            stats={
                "groups": len(sigs),
                "content_merges": sum(1 for a, b in content_pairs),
                "auto_merges": len(auto) - sum(1 for a, b in content_pairs),
                "llm_pending": len(llm),
                "kept": len(keep),
                "merged_files": len({merged[c] for c in sigs}),
            },
        )

    def _is_antipodal_swap(self, a: str, b: str) -> bool:
        """True if the names differ only by one antipodal token swap.

        ``smart_contract_integer_overflow`` vs ``smart_contract_integer_
        underflow`` share 3/4 tokens (word-Dice 0.75) so name similarity alone
        would auto-merge them — but overflow and underflow are different
        vulnerabilities. When the sole differing token is an antipodal pair
        (encode/decode, encrypt/decrypt, ...) the pair must be LLM-judged or
        kept, never silently merged.
        """
        ta, tb = a.split("_"), b.split("_")
        if len(ta) != len(tb):
            return False
        diffs = [i for i in range(len(ta)) if ta[i] != tb[i]]
        if len(diffs) != 1:
            return False
        x, y = ta[diffs[0]], tb[diffs[0]]
        return self.ANTIPODAL_TOKENS.get(x) == y or self.ANTIPODAL_TOKENS.get(y) == x

    def _pick_representative(
        self, members: list[str], sigs: dict[str, _Signature]
    ) -> str:
        """Pick the canonical name a merged group keeps.

        Prefer names that are themselves mapping canonicals (``jwt`` over
        ``jwt_attacks``); otherwise the group with the most tricks (``php_disable
        _functions_bypass`` over the rarer ``disable_functions_bypass``).
        Tie-break: shorter name, then lexicographic.
        """
        pool = [m for m in members if m in self.canonical_values] or members
        # Most tricks, then shortest name, then lexicographically smallest.
        return min(pool, key=lambda m: (-sigs[m].count, len(m), m))

    # -- LLM judge ------------------------------------------------------------

    async def judge_pairs(
        self, candidates: list[tuple[str, str]], sigs: dict[str, _Signature]
    ) -> set[tuple[str, str]]:
        """Ask the LLM which candidate pairs are the same technique.

        Returns the set of pairs judged as merges. Pairs the model says to keep
        separate are stored on :attr:`llm_kept`. Skips silently when no LLM
        client is configured. Batches internally with bounded concurrency; the
        system prompt is static so every batch hits the same cache entry.
        """
        if not self.llm_client or not candidates:
            return set()

        batches = [
            candidates[i : i + self.LLM_BATCH_SIZE]
            for i in range(0, len(candidates), self.LLM_BATCH_SIZE)
        ]
        concurrency = min(settings.llm_max_concurrency, len(batches))
        semaphore = asyncio.Semaphore(concurrency)

        async def _judge_batch(batch: list[tuple[str, str]]) -> tuple[set, list]:
            if not self.llm_client:
                return set(), list(batch)
            user_message = self._build_judge_message(batch, sigs)
            try:
                parsed = await self.llm_client.create_message(
                    _JUDGE_SYSTEM_PROMPT,
                    user_message,
                    cache_system=True,
                )
            except Exception as e:  # noqa: BLE001 (LLM failure -> keep separate)
                logger.warning("LLM judge batch failed, keeping pairs separate: %s", e)
                return set(), list(batch)

            verdicts = self._extract_verdicts(parsed)
            accepted = set()
            kept: list[tuple[str, str]] = []
            # pair_id in the LLM message is 0-based within the batch.
            for pair_id, pair in enumerate(batch):
                if verdicts.get(pair_id) == "merge":
                    accepted.add(pair)
                else:
                    kept.append(pair)
            return accepted, kept

        async def _bounded(batch: list[tuple[str, str]]):
            async with semaphore:
                return await _judge_batch(batch)

        results = await asyncio.gather(*(_bounded(b) for b in batches))
        accepted: set[tuple[str, str]] = set()
        self._llm_kept = []
        for ok, kept in results:
            accepted.update(ok)
            self._llm_kept.extend(kept)
        return accepted

    def _build_judge_message(
        self, batch: list[tuple[str, str]], sigs: dict[str, _Signature]
    ) -> str:
        """Serialize candidate pairs with representative evidence per group."""
        payload = []
        for pair_id, (a, b) in enumerate(batch):
            payload.append(
                {
                    "pair_id": pair_id,
                    "technique_a": self._group_evidence(a, sigs),
                    "technique_b": self._group_evidence(b, sigs),
                }
            )
        return json.dumps(payload)

    def _group_evidence(self, canonical: str, sigs: dict[str, _Signature]) -> dict:
        """Compact evidence for one group: name, top titles, sample description."""
        sig = sigs[canonical]
        top_titles = [
            " ".join(sorted(ts))
            for ts, n in sig.titles.most_common(6)
            for _ in range(min(n, 1))
        ]
        return {
            "name": canonical,
            "trick_count": sig.count,
            "sample_titles": top_titles,
        }

    @staticmethod
    def _extract_verdicts(parsed) -> dict[int, str]:
        """Normalize the LLM JSON into {pair_id: 'merge'|'keep'}."""
        # Accept {"verdicts": [...]} or a bare list.
        data = parsed
        if isinstance(parsed, dict):
            for key in ("verdicts", "results", "pairs"):
                if isinstance(parsed.get(key), list):
                    data = parsed[key]
                    break
            else:
                return {}
        if not isinstance(data, list):
            return {}
        verdicts: dict[int, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            pid = item.get("pair_id", item.get("id"))
            verdict = str(item.get("verdict", "")).lower()
            if isinstance(pid, int) and verdict in ("merge", "keep"):
                verdicts[pid] = verdict
        return verdicts


# The judge system prompt is STATIC (never interpolated with pair data) and
# >= 1024 tokens so it hits Anthropic's ephemeral prompt cache across every
# batch call — project convention (see CLAUDE.md). Keep it that way: any edit
# must keep the text static and at or above ~4096 characters.
_JUDGE_SYSTEM_PROMPT = """\
You are an expert CTF (Capture The Flag) technique deduplication judge. Your job
is to decide, for each pair of candidate techniques, whether they are the SAME
underlying technique (and should be merged into one file) or DISTINCT techniques
(and should stay in separate files). You are the final arbiter on pairs that
automatic name-similarity scoring could not decide with confidence.

CONTEXT — the knowledge base and why we dedup:
We maintain a knowledge base of reusable CTF tricks. Each trick is a short
"how to attack X" note extracted from a real challenge writeup, tagged with a
technique name. Each technique name maps to a markdown file that collects every
trick tagged with it. Because tricks are extracted by an LLM across ~40k
writeups, the SAME attack is often tagged with several slightly different names:
different spellings, synonyms, or long chatty paraphrases. Near-duplicate names
that denote the same attack should be merged into one file so the knowledge base
does not carry two files describing the same technique and so retrieval is not
polluted by near-identical chunks. However, we must NEVER merge genuinely
distinct attacks just because their names happen to look similar or share a word.

WHAT COUNTS AS THE SAME TECHNIQUE — merge these:
- Pure spelling / inflection variants with no meaning change: "php_disable_
  functions_bypass" vs "php_disabled_functions_bypass"; "antidebug" vs
  "anti_debug"; "bruteforce" vs "brute_force"; "xor" vs "xoring".
- Prefix/suffix noise with no meaning change: "hash_cracking" vs
  "hash_cracking_attack"; "jwt" vs "jwt_attack"; "crc32_plaintext_recovery"
  vs "zip_crc32_plaintext_recovery" (same underlying CRC plaintext recovery).
- Synonyms: "local_file_inclusion" vs "lfi"; "type_juggling" vs
  "loose_comparison"; "git_source_disclosure" vs "exposed_git_repository";
  "shellcode" vs "machine_code".
- Chatty rephrasings of one attack: "use_after_free_overwrite" vs "uaf";
  "php_preg_replace_e" vs "php_preg_replace_rce" (the /e modifier RCE).
- A specific sub-technique that is the same attack described more narrowly,
  when it would be better served inside the parent file: "audio_steganography"
  vs "spectrogram_steganography" (both audio stego); "ecb_byte_at_a_time" vs
  "aes_ecb_byte_at_a_time" (AES-ECB is the common instance of ECB byte-at-a-time).

WHAT COUNTS AS DISTINCT TECHNIQUES — DO NOT merge, even if the names look similar:
- Different attack classes or vectors, even sharing a generic word:
  "sql_injection" vs "json_injection" vs "ldap_injection" vs "nosql_injection";
  "command_injection" vs "code_injection" when the injection point genuinely
  differs; "aslr_bypass" (userland) vs "kaslr_bypass" (kernel address-space
  layout); "jsonp_injection" vs "json_injection".
- Different algorithms or modes: "tea_crypto" vs "xtea_crypto"; "aes_cbc_
  oracle" vs "aes_ecb_oracle"; "hash_length_extension" vs "length_extension"
  is a MERGE, but "cbc_bit_flipping" vs "cbc_padding_oracle" are distinct.
- Different targets that happen to share a prefix: "dns_qr_exfiltration" vs
  "dns_txt_exfiltration"; "apng_forensics" vs "png_forensics" (apng is a
  specific animation format); "smart_contract_integer_overflow" vs
  "smart_contract_integer_underflow" (opposite vulnerabilities).
- A general umbrella vs a specific sub-technique that deserves its own file:
  "stegano" (all stego) vs "lsb_steganography" (specific LSB); "authentication_
  bypass" (broad) vs "jwt_none_algorithm" (specific). If you are unsure whether
  an umbrella/subtype pair should merge, lean KEEP — it is safer to keep a
  specific technique discoverable under its own name than to bury it in a broad
  file.

DECISION SIGNALS, strongest first:
1. Identical trick titles across the two groups is very strong evidence they
   describe the same attack. Weigh it heavily.
2. Shared key_code / payload snippets are supporting evidence, but beware
   common payloads ("cat /etc/passwd", "ls -la") that appear in many unrelated
   techniques — they are weak evidence on their own.
3. Name similarity is evidence only when it agrees with content. If the names
   are similar but the sample trick titles point at different attacks, keep.
4. Shared source writeup URLs are NOT evidence: one challenge writeup often
   covers several distinct attacks (SQLi AND XSS AND path traversal).

The input is a JSON array. Each element has a "pair_id" (int), and two objects
"technique_a" and "technique_b", each with "name" (the canonical technique name
to merge under), "trick_count", and "sample_titles" (up to 6 example trick
titles from that group).

Output STRICTLY as a single JSON object, never anything else:
{"verdicts": [{"pair_id": 0, "verdict": "merge"|"keep", "reason": "short justification"}]}
Rules for the output:
- "verdict" must be exactly the lowercase string "merge" or "keep".
- "pair_id" must match the id from the input, and every input pair must appear
  exactly once.
- "reason" should be 1-2 short clauses naming the deciding signal (e.g.
  "same attack, spelling variant" or "different modes: CBC vs ECB").
- Do not include markdown fences, prose, or any other text outside the JSON.
"""


def merged_groups(
    groups: dict[str, list[Trick]], merged_mapping: dict[str, str]
) -> dict[str, list[Trick]]:
    """Regroup tricks by merged canonical names (identity when unmerged)."""
    regrouped: dict[str, list[Trick]] = defaultdict(list)
    for canonical, tricks in groups.items():
        regrouped[merged_mapping.get(canonical, canonical)].extend(tricks)
    return dict(regrouped)


async def run_llm_merges(
    merger: TechniqueMerger,
    result: TechniqueMergeResult,
    sigs: dict[str, _Signature] | None = None,
) -> dict[str, str]:
    """Run the LLM judge on a merge result and fold accepted pairs in.

    Returns the final ``{old_canonical: merged_canonical}`` mapping (after LLM
    merges are unioned in, transitively). Pairs the LLM keeps separate are left
    on ``merger.llm_kept``. ``sigs`` defaults to the one built by
    :meth:`TechniqueMerger.build_merged_mapping`.
    """
    if not result.llm_candidates:
        return result.merged_mapping
    if sigs is None:
        sigs = merger._sigs

    accepted = await merger.judge_pairs(result.llm_candidates, sigs)

    # Union accepted pairs into the existing mapping (transitive).
    final = dict(result.merged_mapping)
    parent = {n: n for n in final}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for name in final:
        parent[name] = name
    for a, b in accepted:
        union(a, b)

    # Representatives were chosen before LLM merges; recompute deterministically
    # using the same preference order (mapping canonical > more tricks > shorter).
    comp: dict[str, list[str]] = defaultdict(list)
    for name in final:
        comp[find(name)].append(name)
    for members in comp.values():
        vals = [m for m in members if m in merger.canonical_values] or members
        rep = min(
            vals,
            key=lambda m: (-(sigs[m].count if m in sigs else 0), len(m), m),
        )
        for m in members:
            final[m] = rep
    return final
