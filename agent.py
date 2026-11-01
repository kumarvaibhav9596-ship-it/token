"""Offline, policy-grounded ZentroMart text-support demo."""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import json
import math
import re
import secrets
import sys
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = SOURCE_ROOT if (SOURCE_ROOT / "data").is_dir() else SOURCE_ROOT.parent
if not (ROOT / "data").is_dir():
    ROOT = SOURCE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import tools as mock_tools
import email_adapter
import llm_adapter
import voice_adapter


MONTHS = {name.lower(): number for number, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"), 1)}
STOP_WORDS = set("""
a an and are as at be been being by can could did do does for from had has have
he her here him his how i if in into is it its me my of on or our please should
so than that the their them then there these they this to us was we were what
when where which who will with would you your tell explain show help need want
zentromart company customer policy policies agent support information details
order orders refund refunds return returns item items delivery shipping payment
days day time come get use bought received get aap mera meri mere hai hain kya
ka ki ke ko mein par se mujhe maine aur kab kitne kitna kya
""".split())
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
SECTION_RE = re.compile(r"^##\s+\[([^\]]+)\]\s*(.*)$", re.MULTILINE)
SKU_RE = re.compile(r"\b[A-Z]{3}-\d{4}\b", re.IGNORECASE)
ORDER_RE = re.compile(r"(?<!\d)(\d{4})[\s-]?(\d{4})(?!\d)")
PIN_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)(?:\s+(20\d{2}))?\b", re.IGNORECASE)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_token(token: str) -> str:
    token = token.lower()
    aliases = {
        "delivery": "deliver", "delivered": "deliver", "delivers": "deliver",
        "shipping": "ship", "shipped": "ship", "ships": "ship",
        "refunds": "refund", "refunded": "refund", "returns": "return",
        "returned": "return", "cancellations": "cancel", "cancelled": "cancel",
        "canceled": "cancel", "electronics": "electronic", "fees": "fee",
        "days": "day", "shoes": "footwear", "shoe": "footwear",
        "warranties": "warranty",
    }
    if token in aliases:
        return aliases[token]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def tokens(text: str) -> list[str]:
    return [_normalize_token(token) for token in TOKEN_RE.findall(text.lower())
            if _normalize_token(token) not in STOP_WORDS]


def _load_fixed_today() -> date:
    meta = (ROOT / "kb" / "00_about_this_knowledge_base.md").read_text(encoding="utf-8")
    match = re.search(r"frozen on\s+\*\*(\d{4}-\d{2}-\d{2})\*\*", meta, re.IGNORECASE)
    return date.fromisoformat(match.group(1)) if match else date(2026, 10, 20)


TODAY = _load_fixed_today()
HOLIDAYS = {date.fromisoformat(row["date"]) for row in read_csv(ROOT / "data" / "holidays.csv")}


@dataclass
class Section:
    section_id: str
    heading: str
    body: str
    doc_id: str
    title: str
    doc_type: str
    status: str
    effective_from: str = ""

    @property
    def searchable_text(self) -> str:
        return f"{self.heading}\n{self.body}"


def load_sections() -> list[Section]:
    sections: list[Section] = []
    for path in sorted((ROOT / "kb").glob("*.md")):
        content = path.read_text(encoding="utf-8")
        front = re.search(r"\A---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        metadata: dict[str, str] = {}
        if front:
            for line in front.group(1).splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    metadata[key.strip()] = value.strip().strip("\"'")
        if metadata.get("status", "current") != "current":
            continue
        matches = list(SECTION_RE.finditer(content))
        if not matches:
            sections.append(Section(
                metadata.get("doc_id", path.stem), path.stem,
                content[front.end():] if front else content,
                metadata.get("doc_id", path.stem), metadata.get("title", path.stem),
                metadata.get("doc_type", "faq"), metadata.get("status", "current"),
                metadata.get("effective_from", "")))
            continue
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            sections.append(Section(
                match.group(1), match.group(2).strip(), content[match.end():end].strip(),
                metadata.get("doc_id", path.stem), metadata.get("title", path.stem),
                metadata.get("doc_type", "faq"), metadata.get("status", "current"),
                metadata.get("effective_from", "")))
    return sections


class SupportAgent:
    def __init__(self, llm_enabled: bool | None = None) -> None:
        self.llm_enabled = llm_adapter.configured() if llm_enabled is None else llm_enabled
        self.sections = load_sections()
        self.products = read_csv(ROOT / "data" / "products.csv")
        self.pincodes = {row["pincode"]: row for row in read_csv(ROOT / "data" / "pincodes.csv")}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.avg_doc_len = sum(len(tokens(s.searchable_text)) for s in self.sections) / max(1, len(self.sections))
        self.document_frequency: Counter[str] = Counter()
        self._embedding_vectors: list[list[float]] | None = None
        self._embedding_lock = threading.Lock()
        for section in self.sections:
            self.document_frequency.update(set(tokens(section.searchable_text)))

    def reset(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)

    def respond(self, session_id: str, message: str) -> dict[str, Any]:
        clean = message.strip()
        if not clean:
            return self._result("Please type a question so I can help.", "clarify")
        if len(clean) > 2000:
            return self._result("That message is too long for this demo. Please send a shorter question.", "clarify")
        with self.lock:
            session = self.sessions.setdefault(session_id, {
                "greeted": False, "pending_order_id": None, "verification_attempts": 0,
                "verified_order": None, "order_issue": "", "refund_method": None,
                "refund_waiting_for": None, "refund_start_date": None,
                "failed_verifications": {}, "policy_history": [],
            })
            escalation_message = self._escalation_message_key(clean)
            if session.get("last_escalation_message") != escalation_message:
                session.pop("last_escalation_signature", None)
                session.pop("last_escalation_message", None)
                session.pop("last_ticket_id", None)
            first_reply = not session["greeted"]
            session["greeted"] = True
            result = self._route(clean, session)
            if first_reply:
                result["answer"] = "Hi, I'm ZentroMart's AI support assistant. " + result["answer"]
            return result

    def _result(self, answer: str, decision: str, sources: list[dict[str, str]] | None = None,
                tool_calls: list[str] | None = None) -> dict[str, Any]:
        return {"answer": answer, "decision": decision, "sources": sources or [],
                "tool_calls": tool_calls or []}

    def _route(self, message: str, session: dict[str, Any]) -> dict[str, Any]:
        lowered = message.lower()
        order_id = self._extract_order_id(message)
        if (session.get("refund_waiting_for")
                and not any(term in lowered for term in ("refund", "money back", "paisa"))
                and not self._refund_method(lowered) and not self._parse_start_date(message)):
            session["refund_waiting_for"] = None
            session["refund_method"] = None
            session["refund_start_date"] = None

        # Urgent policy handovers happen before any order details are disclosed.
        reason = self._urgent_reason(message)
        warranty_question = any(word in lowered for word in ("warranty", "assured", "वारंटी", "अश्योर्ड"))
        product = self._find_product(message, for_warranty=warranty_question)
        if (warranty_question and not product
                and re.search(r"\b(?:sku|model|earbuds?|speaker|smartphone|phone|laptop|watch|charger|"
                              r"power bank|mixer|television|tv)\b", lowered)):
            return self._result("Which exact product or SKU do you mean? Warranty eligibility depends on the catalog listing and seller.",
                                "clarify", self._sources_by_id("MKS-3", "WAR-1", "WAR-2"))
        if not reason and product and self._is_damage_report(message) and int(product.get("price_inr") or 0) > 10000:
            verified_order = session.get("verified_order")
            prior_damage = bool(verified_order and verified_order.get("damage_status")
                                and verified_order.get("damage_sku") == product.get("sku"))
            requests_approval = bool(re.search(r"\bapprove\b|approve kar| मंज़ूरी|मंजूरी", lowered))
            if not prior_damage or requests_approval:
                reason = "Damaged or defective item is worth more than ₹10,000; Priority Resolution must review it."
        if reason:
            verified_order = session.get("verified_order")
            if verified_order and not order_id:
                order_id = str(verified_order.get("order_id") or "") or None
            handoff_order = (verified_order if verified_order
                             and order_id == str(verified_order.get("order_id")) else None)
            return self._create_ticket(message, session, reason, order_id, handoff_order)

        if re.search(r"\b(phone number|email address|home address|street address)\b.*\b(other|someone|person|customer|neighbor|who ordered)\b|\b(person who ordered|another customer|someone else's)\b", lowered):
            return self._result("I can't share another customer's phone number, email, or address. Those details stay private, even when requested.",
                                "refuse", self._sources_by_id("IDV-4", "ESC-4"))

        verified = session.get("verified_order")
        pending_id = session.get("pending_order_id")
        if order_id and verified and order_id == str(verified.get("order_id")):
            return self._handle_verified_order(message, session, verified)
        if order_id and (not pending_id or order_id != pending_id):
            # A new order always starts a fresh verification flow.
            failed_counts = session.setdefault("failed_verifications", {})
            if int(failed_counts.get(order_id, 0)) >= 2:
                session["pending_order_id"] = None
                session["verification_attempts"] = 0
                session["verified_order"] = None
                session["order_issue"] = ""
                return self._result(
                    "I couldn't verify this order, so I can't share its details. Please contact support from the registered phone number.",
                    "refuse")
            session["pending_order_id"] = order_id
            session["verification_attempts"] = int(failed_counts.get(order_id, 0))
            if not session.get("order_issue") or session.get("verified_order"):
                session["order_issue"] = message
            session["verified_order"] = None
            try:
                mock_tools.lookup_order(order_id)
            except Exception:
                return self._result("I couldn't start the order lookup just now. Please try again shortly.", "clarify")
            return self._result(
                f"Please share either the last 4 digits of the registered phone number or the delivery pincode for order {order_id}.",
                "clarify", tool_calls=[f"lookup_order(order_id={order_id})"])
        if pending_id:
            if self._is_verification_attempt(message):
                return self._verify_pending(message, session)
            if self._looks_like_general_policy_question(message):
                # A new general question cancels the unfinished order flow.
                session["pending_order_id"] = None
                session["verification_attempts"] = 0
                session["order_issue"] = ""
                session["refund_waiting_for"] = None
                session["refund_method"] = None
                session["refund_start_date"] = None
            else:
                return self._verify_pending(message, session)
        if "coupon" in lowered and any(term in lowered for term in ("two", "2", "multiple", "combine", "combined", "do coupon")):
            return self._result("Only 1 coupon can be applied per order; coupons can't be combined.",
                                "answer", self._sources_by_id("CPN-1"))
        if verified and self._is_order_followup(message) and not self._looks_like_general_policy_question(message):
            return self._handle_verified_order(message, session, verified)
        if self._needs_order_verification(message):
            session["order_issue"] = message
            return self._result("Could you share the 8-digit order ID?", "clarify")

        if re.search(r"\b(cancel|stop)\b", lowered) and re.search(r"\b(shipped|on its way|on the way)\b", lowered):
            return self._result("A shipped order can't be cancelled. You can refuse the parcel at the door, or accept delivery and return an eligible item within its return window.",
                                "answer", self._sources_by_id("CAN-2"))

        if any(term in lowered for term in ("not at home", "not home", "away when", "not in when", "not be home")):
            return self._result("The courier makes up to 3 delivery attempts. After a missed attempt, you'll get an SMS to reschedule; if the third attempt fails, the parcel returns to the warehouse.",
                                "answer", self._sources_by_id("DEL-3", "REF-1"))

        pin = PIN_RE.search(message)
        if pin and any(word in lowered for word in ("deliver", "delivery", "cod", "express", "pincode", "pin code")):
            pin_row = self.pincodes.get(self._ascii_digits(pin.group(1)))
            if pin_row:
                result = self._pincode_answer(pin_row)
                if not session["greeted"]:
                    result["answer"] = "Hi, I'm ZentroMart's AI support assistant. " + result["answer"]
                return result

        if product and any(word in lowered for word in ("warranty", "assured", "return", "send back", "send it back", "exchange")):
            return self._product_answer(message, product)
        if product and any(word in message for word in ("वापस", "लौट", "रिफंड", "रिफ़ंड", "वारंटी", "अश्योर्ड")):
            return self._product_answer(message, product)

        if self._asks_return_window(message):
            category = self._return_category_from_text(message)
            if category:
                return self._general_return_window_answer(category)

        if any(word in message for word in ("कितने दिन", "कितने दिनों", "लौटाना होगा", "वापस करना")) and not product:
            return self._result("Which item or product category do you mean? The return window depends on the category and delivery date.",
                                "clarify", self._sources_by_id("RET-2"))

        if self._is_explicitly_unanswerable(message):
            return self._result(
                "I couldn't find that in the supplied policies, so I'm not sure. I can connect you to a human support colleague if you'd like.",
                "refuse")

        if re.search(r"\b(dubai|uae|outside india|outside of india|abroad)\b", lowered) and "deliver" in lowered:
            return self._result("ZentroMart delivers only within India, to pincodes listed as serviceable.",
                                "answer", self._sources_by_id("DEL-5"))

        refund_result = self._refund_timeline(message, session)
        if refund_result is not None:
            return refund_result

        # With a configured API key, retrieve semantically and let GPT-4.1
        # compose only from those supplied policy sections. Order, identity,
        # calculation, refusal, and ticket paths above remain deterministic.
        if self.llm_enabled:
            try:
                semantic = self._retrieve_semantic(message)
                if semantic:
                    evidence = [{"id": section.section_id, "title": section.title,
                                 "type": section.doc_type, "status": section.status,
                                 "effective_from": section.effective_from,
                                 "text": section.searchable_text}
                                for section in semantic]
                    history = session.get("policy_history", [])
                    generated = llm_adapter.generate_grounded_answer(
                        self._sanitize_for_llm(message), evidence, history)
                    if generated["supported"] and generated["decision"] == "answer":
                        by_id = {section.section_id: section for section in semantic}
                        cited = [by_id[section_id] for section_id in generated["source_ids"]
                                 if section_id in by_id]
                        if cited:
                            answer = generated["answer"].strip()
                            if not answer:
                                raise llm_adapter.LLMError("The model returned an empty answer.")
                            session.setdefault("policy_history", []).append({
                                "question": self._sanitize_for_llm(message), "answer": answer,
                            })
                            session["policy_history"] = session["policy_history"][-4:]
                            result = self._result(answer, "answer", self._source_list(cited))
                            result["engine"] = "openai_rag"
                            result["model"] = llm_adapter.model_name()
                            return result
                    # The model abstained or cited no retrieved evidence.
                    result = self._result(generated["answer"], generated["decision"])
                    result["engine"] = "openai_rag"
                    result["model"] = llm_adapter.model_name()
                    return result
            except llm_adapter.LLMError:
                # Keep the existing offline answer available when a provider
                # is unreachable. Its engine label remains explicitly distinct.
                pass

        retrieved = self._retrieve(message)
        if not retrieved:
            result = self._result(
                "I couldn't confirm that from the supplied ZentroMart policies. I can help with a policy question, a verified order lookup, or a human handover.",
                "refuse")
            result["engine"] = "fallback"
            return result
        best, sources = retrieved[0], self._source_list(retrieved)
        excerpt = self._excerpt(best, message)
        result = self._result(
            f"{excerpt} I used the current {best.title} section {best.section_id} as the source.",
            "answer", sources)
        result["engine"] = "rag"
        return result

    @staticmethod
    def _sanitize_for_llm(message: str) -> str:
        """Remove common account identifiers before policy text leaves the app."""
        safe = re.sub(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)|(?<!\d)(?:\d[ -]?){7}\d(?!\d)",
                      "[order or phone number removed]", message)
        safe = re.sub(r"(?<!\d)\d{6}(?!\d)", "[pincode removed]", safe)
        safe = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[email removed]", safe, flags=re.I)
        safe = re.sub(r"(?i)((?:phone|mobile|pincode|pin code|verification|last four digits)[^\d]{0,20})\d{4,6}",
                      r"\1[verification value removed]", safe)
        return safe

    def _retrieve_semantic(self, query: str, limit: int = 5) -> list[Section]:
        """Embed KB sections once per process and rank query vectors by cosine."""
        if not self.sections:
            return []
        with self._embedding_lock:
            if self._embedding_vectors is None:
                self._embedding_vectors = llm_adapter.create_embeddings(
                    [f"{section.title}\n[{section.section_id}]\nstatus:{section.status}; "
                     f"effective_from:{section.effective_from}\n{section.searchable_text}"
                     for section in self.sections])
            vectors = self._embedding_vectors
        query_vector = llm_adapter.create_embeddings([self._sanitize_for_llm(query)])[0]
        ranked = [(llm_adapter.cosine_similarity(query_vector, vector), section)
                  for section, vector in zip(self.sections, vectors)]
        ranked.sort(key=lambda item: (item[0],
                     {"policy": 5, "agent_guide": 4, "help": 3,
                      "faq": 2, "meta": 1, "marketing": 0}.get(item[1].doc_type, 0)), reverse=True)
        if not ranked or ranked[0][0] < 0.18:
            return []
        best = ranked[0][0]
        relevant = [(score, section) for score, section in ranked
                    if score >= max(0.18, best - 0.12)]
        # If a current policy is nearly as relevant as another source, the
        # supplied precedence rules require the policy to be the authority.
        policies = [(score, section) for score, section in relevant
                    if section.doc_type == "policy" and score >= best - 0.08]
        if policies:
            relevant = policies
        return [section for _score, section in relevant[:limit]]

    def _extract_order_id(self, message: str) -> str | None:
        # Don't treat a phone number, pincode, or verification value as an order ID.
        if re.search(r"phone|mobile|pincode|pin\s*code", message, re.IGNORECASE):
            return None
        match = ORDER_RE.search(message)
        return self._ascii_digits(f"{match.group(1)}{match.group(2)}") if match else None

    def _needs_order_verification(self, message: str) -> bool:
        text = message.lower()
        cues = (
            "where is my order", "track my order", "track order", "order status",
            "status of my order", "help with my order", "about my order", "this order",
            "my package", "my parcel", "my refund hasn't", "my refund has not",
            "refund not arrived", "refund hasn't arrived", "refund not received",
            "help with an order", "order ke baare mein", "order ke baare me",
            "मुझे अपने ऑर्डर", "अपने ऑर्डर के बारे में", "ऑर्डर के बारे में",
            "ऑर्डर के बारे मे", "मुझे ऑर्डर के बारे में", "my order hasn't arrived",
            "my order has not arrived", "refund not received", "where is it",
        )
        return any(cue in text for cue in cues)

    @staticmethod
    def _is_explicitly_unanswerable(message: str) -> bool:
        return bool(re.search(
            r"\bmidnight\b|\bstudent discount\b|\bbulk order\b|\bgst invoice\b|"
            r"\bshop(?:s|ping)? near me\b|\bstore near me\b|\bpick up from (?:a )?store\b|"
            r"\bpickup from (?:a )?store\b|दुकान|शॉप",
            message, re.IGNORECASE))

    def _is_order_followup(self, message: str) -> bool:
        text = message.lower()
        phrases = (
            "this order", "where is", "track", "wrong item", "third time", "three times",
            "teesri baar", "teesri bar", "तीसरी बार", "वापस", "लौट", "रिफंड", "रिफ़ंड",
            "कहाँ", "कहा", "मेरा ऑर्डर",
        )
        words = ("it", "its", "return", "refund", "delivery", "status", "cancel",
                 "damaged", "broken", "track", "ship", "voucher", "compensation",
                 "late", "express")
        return (any(phrase in text for phrase in phrases)
                or any(re.search(rf"\b{re.escape(word)}\b", text) for word in words))

    @staticmethod
    def _looks_like_general_policy_question(message: str) -> bool:
        text = message.lower()
        if re.search(r"\b(my|this|that)\s+(?:order|refund|delivery|parcel|item)\b|"
                     r"\b(?:order|parcel|item)\s+(?:id|number)\b|\bwhere is (?:it|my)\b", text):
            return False
        if any(phrase in text for phrase in ("new question", "different question", "never mind", "forget that")):
            return True
        has_topic = any(re.search(rf"\b{word}\b", text) for word in (
            "return", "refund", "delivery", "shipping", "warranty", "coupon", "payment", "pincode"))
        asks_general = bool(re.search(r"\b(?:policy|in general|how long|how many days|what is|what are|"
                                      r"can i|do you|does zentromart|tell me about)\b", text))
        category = bool(re.search(r"\b(?:shoes?|footwear|clothing|fashion|electronics|books?|"
                                  r"beauty|grocery|home and kitchen)\b", text))
        return ("policy" in text or (has_topic and asks_general) or (category and has_topic))

    @staticmethod
    def _extract_verification_proof(message: str) -> str | None:
        digits = "".join(ch for ch in message if ch.isdigit())
        explicit = re.search(
            r"(?:last\s*(?:four|4)|आख़िरी\s*(?:चार|4)|आखिरी\s*(?:चार|4)|अंतिम\s*(?:चार|4)|"
            r"phone|mobile|pincode|pin\s*code)[^0-9]{0,20}(\d{4}|\d{6})\b",
            message, re.IGNORECASE)
        if explicit:
            return SupportAgent._ascii_digits(explicit.group(1))
        if re.fullmatch(r"\s*\d{4}\s*|\s*\d{6}\s*", message):
            return SupportAgent._ascii_digits(digits)
        return None

    @staticmethod
    def _ascii_digits(value: str) -> str:
        return "".join(str(unicodedata.digit(ch)) for ch in value if ch.isdigit())

    @classmethod
    def _is_verification_attempt(cls, message: str) -> bool:
        return (cls._extract_verification_proof(message) is not None
                or bool(re.search(r"\b(?:phone|mobile|pincode|pin\s*code|verification|last\s*(?:four|4))\b",
                                  message, re.IGNORECASE))
                or bool(re.fullmatch(r"\s*\d+\s*", message)))

    def _verify_pending(self, message: str, session: dict[str, Any]) -> dict[str, Any]:
        digits = "".join(ch for ch in message if ch.isdigit())
        if len(digits) == 10 and re.search(r"phone|mobile", message, re.IGNORECASE):
            return self._result("For privacy, please send only the last 4 phone digits or the 6-digit delivery pincode.", "clarify")
        proof = self._extract_verification_proof(message)
        if not proof:
            return self._result("Please send one verification value: the last 4 registered phone digits or the delivery pincode.", "clarify")
        order_id = session["pending_order_id"]
        try:
            order = mock_tools.lookup_order(order_id, proof)
        except Exception:
            return self._result("I couldn't check that verification value just now. Please try again shortly.", "clarify")
        if not order.get("verified"):
            failed_counts = session.setdefault("failed_verifications", {})
            failed_counts[order_id] = int(failed_counts.get(order_id, 0)) + 1
            session["verification_attempts"] = failed_counts[order_id]
            if failed_counts[order_id] < 2:
                return self._result("That didn't match. Please retry once with the last 4 registered phone digits or the delivery pincode.", "clarify",
                                    tool_calls=["lookup_order(verification supplied)"])
            session["pending_order_id"] = None
            session["order_issue"] = ""
            return self._result("I couldn't verify this order, so I can't share its details. Please contact support from the registered phone number.", "refuse",
                                tool_calls=["lookup_order(verification supplied)"])
        session["pending_order_id"] = None
        session["verification_attempts"] = 0
        session.setdefault("failed_verifications", {}).pop(order_id, None)
        session["verified_order"] = order
        return self._handle_verified_order(session.get("order_issue") or message, session, order,
                                           tool_calls=["lookup_order(verification supplied)"])

    def _handle_verified_order(self, message: str, session: dict[str, Any], order: dict[str, Any],
                               tool_calls: list[str] | None = None) -> dict[str, Any]:
        order_id = str(order.get("order_id", ""))
        if int(order.get("contacts_last_7_days") or 0) >= 2 and (self._repeat_contact_claim(message) or self._is_specific_order_issue(message)):
            return self._create_ticket(message, session,
                                       "Third or later contact about the same issue within 7 days.",
                                       order_id, order, tool_calls)
        if any(term in message.lower() for term in ("refund", "रिफंड", "रिफ़ंड")) and self._refund_is_overdue(order):
            return self._create_ticket(message, session,
                                       "Refund is more than 3 business days past its maximum policy timeline.",
                                       order_id, order, tool_calls)
        damage_status = str(order.get("damage_status") or "").lower()
        asks_for_approval = bool(re.search(r"\bapprove\b|approve kar|मंज़ूरी|मंजूरी", message.lower()))
        asks_for_damage_status = bool(re.search(
            r"\b(?:already|status|update|progress)\b|where is the replacement", message.lower()))
        if self._is_damage_report(message) and damage_status and asks_for_damage_status and not asks_for_approval:
            return self._damage_status_answer(order, tool_calls)
        damage_item = self._damaged_order_item(order)
        if self._is_damage_report(message) and damage_item and int(damage_item.get("unit_price_inr") or 0) > 10000:
            return self._create_ticket(message, session,
                                       "Damaged or defective item is worth more than ₹10,000; Priority Resolution must review it.",
                                       order_id, order, tool_calls)
        if self._is_damage_report(message):
            return self._result(
                "Please report damage within 48 hours of delivery and attach photos of the item and packaging. After verification, the policy allows a free replacement or full refund; I can't approve it here.",
                "answer", self._source_list(self._retrieve("damaged item report photos 48 hours replacement refund")),
                tool_calls)
        text = message.lower()
        late_express = "express" in text and ("late" in text or "delay" in text)
        asks_compensation = any(term in text for term in ("voucher", "compensation", "300 rupee", "300 rupaye", "rupees 300", "₹300", "additional credit", "extra credit"))
        if late_express or asks_compensation and any(term in text for term in ("late", "delay", "express")):
            if not self._order_was_late(order):
                answer = "The verified order record doesn't show delivery after its promised date, so I can't confirm a late-delivery credit."
            elif str(order.get("delivery_type", "")).lower() == "express" and not order.get("is_plus"):
                credited = order.get("express_late_credit_inr")
                credited_on = self._format_date(order.get("express_late_credit_on"))
                if credited:
                    answer = f"The order record shows ₹{credited} Express-fee credit to Wallet"
                    answer += f" on {credited_on}." if credited_on else "."
                else:
                    answer = "For a late Express order, the ₹79 Express fee paid is credited automatically to Wallet within 48 hours. The policy doesn't allow an additional ₹300 voucher."
            elif str(order.get("delivery_type", "")).lower() == "express":
                answer = "ZentroMart Plus members pay no Express fee, so the policy provides no late-fee credit or additional voucher."
            else:
                answer = "The policy gives no compensation for late Standard delivery, and it doesn't allow an additional ₹300 voucher."
            if asks_compensation:
                answer = "I'm sorry, " + answer[0].lower() + answer[1:]
            return self._result(answer, "refuse" if asks_compensation else "answer", self._sources_by_id("DEL-4"), tool_calls)
        if any(term in text for term in ("return", "send back", "send it back", "exchange", "वापस", "लौट")):
            return self._verified_return_answer(order, message, tool_calls)
        if any(term in text for term in ("refund", "money back", "रिफंड", "रिफ़ंड")):
            return self._verified_refund_answer(order, tool_calls)
        if "ship" in text or "शिप" in message:
            if order.get("shipped_on"):
                return self._result(f"I verified order {order_id}. It shipped on {self._format_date(order['shipped_on'])} ({str(order['shipped_on'])[:10]}).", "answer", tool_calls=tool_calls)
            return self._result(f"I verified order {order_id}; the order record does not show a ship date yet.", "answer", tool_calls=tool_calls)
        if "cancel" in text or "कैंसिल" in message:
            status = str(order.get("status") or "").replace("_", " ")
            if order.get("shipped_on"):
                item = (order.get("items") or [None])[0]
                return_window = "the applicable category return window"
                if item:
                    days, _description = self._return_window_data(item.get("return_category") or "", order)
                    if days:
                        return_window = f"{days} days"
                return self._result(
                    f"I verified order {order_id}; it has shipped, so you can't cancel it. You can refuse the parcel at the door; for prepaid orders, the refund usually starts 7 to 10 calendar days after the last delivery attempt once the parcel returns. If you accept it, an eligible item may be returned within {return_window}.",
                    "refuse", self._sources_by_id("CAN-2", "REF-1", "RET-2"), tool_calls)
            return self._result(
                f"I verified order {order_id}; it has not shipped yet. Please cancel it from My Orders in the app; I can't cancel orders for you.",
                "answer", self._source_list(self._retrieve("cancel unshipped order My Orders")), tool_calls)
        status = str(order.get("status") or "not listed").replace("_", " ")
        promised_iso = str(order.get("promised_date") or "")[:10]
        promised = self._format_date(promised_iso)
        latest = self._latest_timeline(order.get("timeline") or "")
        detail = f"Its status is {status}"
        if promised:
            detail += f" and the promised delivery date is {promised} ({promised_iso})"
        if latest:
            detail += f". The latest timeline entry is {latest}"
        return self._result(f"I verified order {order_id}. {detail}.", "answer", tool_calls=tool_calls)

    def _damage_status_answer(self, order: dict[str, Any],
                              tool_calls: list[str] | None = None) -> dict[str, Any]:
        status = str(order.get("damage_status") or "").lower()
        hours = str(order.get("damage_hours_after_delivery") or "").strip()
        if status == "approved_replacement":
            answer = "The verified order record shows the damaged-item claim was approved for replacement. It doesn't list a replacement shipment date."
            sources = self._sources_by_id("DWM-2")
        elif status == "awaiting_photos":
            answer = "The verified order record shows the damage claim is waiting for photos of the item and packaging."
            sources = self._sources_by_id("DWM-1")
        elif status == "priority_resolution":
            answer = "The verified order record shows the claim is with Priority Resolution for human review. I can't promise an outcome or time."
            sources = self._sources_by_id("DWM-3", "ESC-1", "ESC-2")
        elif status == "handled_as_return":
            answer = "The verified order record shows the damage report was handled as a return. Any refund follows the return pickup, quality-check, and payment timelines in the supplied policies."
            sources = self._sources_by_id("DWM-4", "RET-7", "REF-2")
        elif status == "rejected_late_report":
            timing = f" The report was recorded {hours} hours after delivery." if hours else ""
            answer = ("The verified order record shows the damage report was marked rejected as late."
                      f"{timing} Under the policy, reports after 48 hours follow the normal return window if eligible; otherwise, use any applicable brand or ZentroMart Assured warranty.")
            sources = self._sources_by_id("DWM-4", "RET-2", "WAR-1", "WAR-2")
        else:
            return self._result("The verified order record lists a damage report, but its current handling status isn't available. I can explain the damage policy or connect you with a human colleague.",
                                "clarify", self._sources_by_id("DWM-1", "ESC-1"), tool_calls)
        return self._result(answer, "answer", sources, tool_calls)

    def _verified_return_answer(self, order: dict[str, Any], message: str,
                                tool_calls: list[str] | None) -> dict[str, Any]:
        items = order.get("items") or []
        if not items:
            return self._result("I verified the order, but its item details aren't available in the lookup. I can still explain the general return policy.", "clarify", tool_calls=tool_calls)
        if self._ambiguous_order_item(items, message):
            choices = ", ".join(str(item.get("title") or "an item") for item in items[:4])
            return self._result(f"I verified the order. Which item should I check: {choices}?", "clarify",
                                tool_calls=tool_calls)
        item = self._choose_order_item(items, message)
        category = item.get("return_category") or "unknown"
        days, window = self._return_window_data(category, order)
        answer = f"I verified the order. {item.get('title', 'This item')}: {window}."
        delivered = order.get("delivered_on")
        if days and delivered:
            try:
                deadline = date.fromisoformat(str(delivered)[:10]) + timedelta(days=days)
                if TODAY <= deadline:
                    answer += f" You can request the return through {deadline.isoformat()}."
                else:
                    answer += f" The return window ended on {deadline.isoformat()}."
            except ValueError:
                pass
        if category == "fashion" and not order.get("is_plus") and not self._is_damage_report(message):
            answer += " A ₹49 reverse-pickup fee applies to change-of-mind returns."
        if category in {"grocery", "innerwear", "digital"}:
            answer += " Damaged, expired, wrong, or missing items follow a separate policy."
        else:
            answer += " The item's condition rules also apply."
        sources = self._sources_by_id("RET-2", "RET-4")
        if category == "electronics":
            sources += self._sources_by_id("RET-3")
        if category == "fashion":
            sources += self._sources_by_id("RET-6", "RET-8")
        if category in {"grocery", "innerwear", "digital"}:
            sources += self._sources_by_id("RET-5")
        sources.append({"id": f"products.csv:{item.get('sku', '')}", "title": item.get("title", "Product catalog")})
        return self._result(answer, "answer", sources, tool_calls)

    def _return_window_data(self, category: str, order: dict[str, Any]) -> tuple[int | None, str]:
        placed = str(order.get("placed_at") or "")[:10]
        delivered = str(order.get("delivered_on") or "")[:10]
        if category == "electronics":
            days = 10 if delivered and delivered < "2026-08-01" else 7
            return days, f"the return window is {days} calendar days from delivery"
        if category == "fashion":
            festive = placed and "2026-10-01" <= placed <= "2026-10-15"
            days = 30 if festive else 15
            return days, f"the return window is {days} calendar days from delivery"
        windows = {
            "home_kitchen": (10, "the return window is 10 calendar days from delivery"),
            "beauty_sealed": (7, "the return window is 7 calendar days from delivery if unopened with the seal intact"),
            "books_media": (10, "the return window is 10 calendar days from delivery"),
            "grocery": (None, "this item isn't returnable for change of mind"),
            "innerwear": (None, "this item isn't returnable for change of mind"),
            "digital": (None, "this item isn't returnable for change of mind"),
        }
        return windows.get(category, (None, "the return window depends on the product category"))

    def _verified_refund_answer(self, order: dict[str, Any],
                                tool_calls: list[str] | None) -> dict[str, Any]:
        fields = (
            ("return_refund_initiated_on", "return_refund_method", "return_refund_amount_inr", "return_rrn", "return_status"),
            ("cancellation_refund_initiated_on", "cancellation_refund_method", "cancellation_refund_amount_inr", "cancellation_rrn", "status"),
            ("rto_refund_initiated_on", "rto_refund_method", "rto_refund_amount_inr", "rto_rrn", "status"),
        )
        for started_key, method_key, amount_key, rrn_key, status_key in fields:
            if order.get(started_key):
                method = str(order.get(method_key) or "the original payment method").replace("_", " ")
                amount = order.get(amount_key)
                rrn = order.get(rrn_key)
                answer = f"I verified the order. Its refund was initiated on {self._format_date(order[started_key])} via {method}"
                if amount:
                    answer += f" for ₹{amount}"
                answer += "."
                if rrn:
                    answer += f" The refund reference is {rrn}."
                return self._result(answer, "answer", self._source_list(self._retrieve("refund reference number timeline")), tool_calls)
        method = str(order.get("return_refund_method") or order.get("payment_method") or "").lower()
        timeline = {
            "upi": "1 to 3 business days",
            "card": "5 to 7 business days",
            "netbanking": "3 to 5 business days",
            "net_banking": "3 to 5 business days",
            "wallet": "instant, within 2 hours",
            "cod_wallet": "instant, within 2 hours",
            "cod_bank": "5 to 7 business days after bank details are verified",
        }.get(method, "the timeline listed for the original payment method")
        if order.get("return_status"):
            if order.get("is_plus"):
                detail = "For ZentroMart Plus, the return refund is initiated at pickup."
            else:
                detail = "After pickup, the item goes through a quality check; once it passes, the refund is initiated within 2 business days."
            answer = f"I verified the order. {detail} After initiation, {method.replace('_', ' ')} refunds take {timeline}."
            return self._result(answer, "answer", self._sources_by_id("RET-7", "REF-2", "PLS-2"), tool_calls)
        answer = f"I verified the order. After a return, the refund is initiated after pickup and any required quality check. This order uses {method.replace('_', ' ')}; refunds take {timeline} after initiation."
        if order.get("is_plus"):
            answer = f"I verified the order. As a ZentroMart Plus member, the return refund is initiated at pickup. This order uses {method.replace('_', ' ')}; refunds take {timeline} after initiation."
        return self._result(answer, "answer", self._sources_by_id("RET-7", "REF-2", "PLS-2"), tool_calls)

    def _refund_is_overdue(self, order: dict[str, Any]) -> bool:
        options = (
            ("return_refund_initiated_on", "return_refund_completed_on", "return_refund_method"),
            ("cancellation_refund_initiated_on", "", "cancellation_refund_method"),
            ("rto_refund_initiated_on", "", "rto_refund_method"),
        )
        maximum = {"upi": 3, "card": 7, "netbanking": 5, "net banking": 5,
                   "wallet": 0, "cod_wallet": 0, "cod_bank": 7, "bank transfer": 7}
        for started_key, completed_key, method_key in options:
            started = order.get(started_key)
            if not started or (completed_key and order.get(completed_key)):
                continue
            method = str(order.get(method_key) or order.get("payment_method") or "").lower().replace(" ", "")
            max_days = maximum.get(method)
            if max_days is None:
                max_days = maximum.get(method.replace("_", " "))
            if max_days is None:
                continue
            try:
                start_date = date.fromisoformat(str(started)[:10])
            except ValueError:
                continue
            elapsed = 0
            day = start_date
            while day < TODAY:
                day += timedelta(days=1)
                if day.weekday() < 6 and day not in HOLIDAYS:
                    elapsed += 1
            return elapsed > max_days + 3
        return False

    def _damaged_order_item(self, order: dict[str, Any]) -> dict[str, Any] | None:
        sku = order.get("damage_sku")
        for item in order.get("items") or []:
            if sku and item.get("sku") == sku:
                return item
        return None

    @staticmethod
    def _order_was_late(order: dict[str, Any]) -> bool:
        try:
            promised = date.fromisoformat(str(order.get("promised_date"))[:10])
            delivered = date.fromisoformat(str(order.get("delivered_on"))[:10])
            return delivered > promised
        except (TypeError, ValueError):
            return False

    def _choose_order_item(self, items: list[dict[str, Any]], message: str) -> dict[str, Any]:
        query = set(tokens(message))
        scored = [(len(query & set(tokens(str(item.get("title", ""))))), item) for item in items]
        return max(scored, key=lambda pair: pair[0])[1]

    def _ambiguous_order_item(self, items: list[dict[str, Any]], message: str) -> bool:
        if len(items) < 2:
            return False
        sku_match = SKU_RE.search(message)
        if sku_match:
            return not any(str(item.get("sku", "")).casefold() == sku_match.group(0).casefold()
                           for item in items)
        query = set(tokens(message))
        scored = [(len(query & set(tokens(str(item.get("title", ""))))), item) for item in items]
        best = max(score for score, _item in scored)
        if best == 0:
            return True
        winners = {str(item.get("sku", "")) for score, item in scored if score == best}
        return len(winners) > 1

    def _urgent_reason(self, message: str) -> str | None:
        text = message.lower()
        if re.search(r"\b(unauthori[sz]ed|fraud|never placed|never ordered|did(?:n['’]?t| not) (?:place|make)|someone (?:used|placed|charged)|not my order|not my transaction)\b", text):
            return "Customer reports an order, payment, or account activity they did not make."
        if any(phrase in text for phrase in ("इंसान से बात", "मानव एजेंट", "किसी व्यक्ति से", "इंसान से जुड़")):
            return "Customer asked to speak with a human support agent."
        if re.search(r"\b(legal action|consumer court|police complaint|sue you|court case)\b", text):
            return "Customer mentioned legal action, a consumer court, or a police complaint."
        if re.search(r"\b(swelling battery|swollen battery|burning smell|sparks?|smoke|caught fire|injur(?:y|ed)|electric shock)\b", text):
            return "Product safety risk requires immediate human handover."
        if re.search(r"\b(talk to|speak to|connect me to|want|need|request) (?:a )?(?:human|person|representative|manager|live agent)\b|\bhuman agent\b|\blive agent\b", text):
            return "Customer asked to speak with a human support agent."
        return None

    @staticmethod
    def _is_damage_report(message: str) -> bool:
        return bool(re.search(r"\b(damag(?:e|ed)|defect(?:ive)?|broken|crack(?:ed)?|wrong item|missing item|torn|leak(?:ing)?)\b", message, re.IGNORECASE))

    @staticmethod
    def _repeat_contact_claim(message: str) -> bool:
        return bool(re.search(r"\b(third time|3rd time|three times|third contact|again and again)\b|teesri baar|teesri bar|तीसरी बार|तीसरी दफ़ा", message, re.IGNORECASE))

    @staticmethod
    def _is_specific_order_issue(message: str) -> bool:
        text = message.lower()
        return any(term in text for term in (
            "refund", "return", "delivery", "where is", "cancel", "damage", "damaged",
            "wrong item", "missing item", "replacement", "broken", "defective", "ship",
            "वापस", "लौट", "रिफंड", "रिफ़ंड", "कहाँ", "कहा", "कैंसिल", "शिप"))

    def _create_ticket(self, message: str, session: dict[str, Any], reason: str,
                       order_id: str | None = None, order: dict[str, Any] | None = None,
                       prior_tool_calls: list[str] | None = None) -> dict[str, Any]:
        detected_intent = "damaged_item" if self._is_damage_report(message) else "human_handover"
        checks = "No order details were disclosed."
        if order:
            checks = f"Order lookup was identity-verified; recorded contacts in the last 7 days: {order.get('contacts_last_7_days', 0)}."
        safe_message = self._redact_ticket_text(message)
        summary = (f"Issue: {safe_message}\nDetected intent: {detected_intent}\n"
                   f"Order ID: {order_id or 'not provided'}\nChecks made: {checks}\n"
                   f"Escalation reason: {reason}")
        signature = (self._escalation_message_key(message), order_id or "", reason, detected_intent)
        existing_id = session.get("last_ticket_id")
        if existing_id and session.get("last_escalation_signature") == signature:
            return self._result(
                f"This request is already logged under ticket {existing_id}. A human support colleague will review it; I can't promise a specific outcome or time.",
                "escalate", self._ticket_sources(reason), list(prior_tool_calls or []))
        try:
            ticket = mock_tools.create_ticket(summary, order_id=order_id, reason=reason, intent=detected_intent)
        except Exception:
            ticket = {"created": False}
        calls = list(prior_tool_calls or []) + ["create_ticket"]
        if ticket.get("created"):
            sources = self._ticket_sources(reason)
            session["last_escalation_signature"] = signature
            session["last_escalation_message"] = self._escalation_message_key(message)
            session["last_ticket_id"] = ticket["ticket_id"]
            answer = f"I created ticket {ticket['ticket_id']}. A human support colleague will review your request; I can't promise a specific outcome or time."
            if order and "refund" in reason.lower():
                rrn = order.get("return_rrn") or order.get("cancellation_rrn") or order.get("rto_rrn")
                if rrn:
                    answer += f" Your refund reference number is {rrn}."
            return self._result(answer, "escalate", sources, calls)
        return self._result("I couldn't create a handover ticket just now. I haven't marked the handover as complete; please try again shortly.",
                            "escalate", tool_calls=calls)

    @staticmethod
    def _escalation_message_key(message: str) -> str:
        safe_message = SupportAgent._redact_ticket_text(message)
        return re.sub(r"\s+", " ", safe_message.strip().casefold())

    @staticmethod
    def _redact_ticket_text(message: str) -> str:
        safe = re.sub(
            r"\b(?:(?:home|street|delivery|billing)\s+)?address\b\s*(?:(?:is|at)\s+|[:=]\s*)?[^\n.;]+",
            "[redacted address]", message, flags=re.IGNORECASE)
        safe = re.sub(
            r"\b\d{1,6}\s+(?:[\w'.-]+\s+){0,4}(?:street|st\.?|road|rd\.?|avenue|ave\.?|"
            r"lane|ln\.?|drive|dr\.?|boulevard|blvd\.?|apartment|flat)\b[^\n.;]*",
            "[redacted address]", safe, flags=re.IGNORECASE)
        safe = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[redacted email]", safe, flags=re.IGNORECASE)
        safe = re.sub(r"(?<!\d)(?:\+91[\s-]?)?\d{10}(?!\d)", "[redacted phone number]", safe)
        # Four- and six-digit values are accepted verification secrets. Keep the
        # eight-digit order ID for the human handover, but remove those secrets.
        skus: list[str] = []
        safe = SKU_RE.sub(lambda match: skus.append(match.group(0)) or f"__SKU_{len(skus) - 1}__", safe)
        safe = re.sub(r"(?<!\d)(?:\d{4}|\d{6})(?!\d)", "[redacted verification value]", safe)
        for index, sku in enumerate(skus):
            safe = safe.replace(f"__SKU_{index}__", sku)
        return safe

    def _ticket_sources(self, reason: str) -> list[dict[str, str]]:
        sources = self._sources_by_id("ESC-1", "ESC-2")
        if "10,000" in reason:
            sources += self._sources_by_id("DWM-3")
        if "safety" in reason.lower():
            sources += self._sources_by_id("DWM-5")
        if "refund" in reason.lower():
            sources += self._sources_by_id("REF-2")
        return sources

    def _find_product(self, message: str, for_warranty: bool = False) -> dict[str, Any] | None:
        sku_match = SKU_RE.search(message)
        if sku_match:
            sku = sku_match.group(0).upper()
            for product in self.products:
                if product["sku"].upper() == sku:
                    selected = dict(product)
                    selected["_matched_skus"] = [product["sku"]]
                    return selected
        exact_titles = [product for product in self.products
                        if str(product.get("title") or "").casefold() in message.casefold()]
        if len(exact_titles) == 1:
            return exact_titles[0]
        query = set(tokens(message))
        if for_warranty:
            query -= {"warranty", "assured", "brand", "claim", "seller", "covered"}
        if len(query) < 2:
            return None
        ranked: list[tuple[float, dict[str, str]]] = []
        for product in self.products:
            title = set(tokens(product["title"]))
            overlap = len(query & title)
            if not overlap:
                continue
            score = overlap / max(2, min(len(title), len(query)))
            ranked.append((score, product))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        if ranked and ranked[0][0] >= 0.55:
            best_score = ranked[0][0]
            tied = [product for score, product in ranked if abs(score - best_score) < 1e-9]
            if len(tied) == 1:
                return tied[0]
            facts = {(row.get("return_category"), row.get("assured"), row.get("warranty_months")) for row in tied}
            if len(facts) == 1 or (not for_warranty and len({row.get("return_category") for row in tied}) == 1):
                selected = dict(tied[0])
                selected["_matched_skus"] = [row["sku"] for row in tied]
                return selected
        return None

    def _product_answer(self, message: str, product: dict[str, str]) -> dict[str, Any]:
        text = message.lower()
        sku = product["sku"]
        matched_skus = product.get("_matched_skus", [sku])
        matched_rows = {row["sku"]: row for row in self.products}
        sources: list[dict[str, str]] = [
            {"id": f"products.csv:{matched_sku}", "title": matched_rows.get(matched_sku, product).get("title", product["title"])}
            for matched_sku in matched_skus
        ]
        if "warranty" in text or "assured" in text or "वारंटी" in message or "अश्योर्ड" in message:
            brand_months = product.get("warranty_months") or "not listed"
            marketplace = str(product.get("fulfilled_by") or "").strip().lower() not in {
                "zentromart", "zentromart warehouse", "zentromart fulfillment", "zentromart fulfilment"
            }
            if marketplace:
                sources += self._sources_by_id("MKS-3", "WAR-1")
                answer = (f"{product['title']} is fulfilled by marketplace seller {product.get('fulfilled_by')}. "
                          f"Marketplace items are not covered by ZentroMart Assured, even if the listing says so; "
                          f"the catalog lists a {brand_months}-month brand warranty.")
            elif product.get("assured", "False").lower() == "true":
                sources += self._sources_by_id("WAR-1", "WAR-2")
                answer = (f"The catalog marks {product['title']} as ZentroMart Assured: it has a 12-month Assured warranty from delivery, in addition to its {brand_months}-month brand warranty. Claims are raised in My Orders in the app.")
            else:
                sources += self._sources_by_id("WAR-1")
                answer = f"The catalog does not mark {product['title']} as ZentroMart Assured; its listed brand warranty is {brand_months} months."
            return self._result(answer, "answer", sources)
        category = product.get("return_category", "")
        windows = {
            "electronics": "7 calendar days from delivery; items delivered before 1 August 2026 keep the earlier 10-day window",
            "fashion": "15 calendar days from delivery; orders placed 1–15 October 2026 have a 30-day festive-sale window",
            "home_kitchen": "10 calendar days from delivery",
            "beauty_sealed": "7 calendar days from delivery, if unopened with the seal intact",
            "books_media": "10 calendar days from delivery",
            "grocery": "not returnable for change of mind",
            "innerwear": "not returnable for change of mind",
            "digital": "not returnable for change of mind",
        }
        window = windows.get(category, "not listed in the product catalog")
        if category in {"grocery", "innerwear", "digital"}:
            sources += self._sources_by_id("RET-2", "RET-5")
        elif category == "electronics":
            sources += self._sources_by_id("RET-2", "RET-3", "RET-4")
        elif category == "fashion":
            sources += self._sources_by_id("RET-2", "RET-6", "RET-8")
        elif category == "beauty_sealed":
            sources += self._sources_by_id("RET-2", "RET-4")
        else:
            sources += self._sources_by_id("RET-2", "RET-4")
        fee_question = any(term in text for term in ("fee", "charged", "pickup"))
        if fee_question:
            sources += self._sources_by_id("RET-6")
        if category == "beauty_sealed" and (re.search(r"\b(opened|open|tried it once)\b", text) or "खोल" in message):
            answer = f"The policy allows {product['title']} to be returned only if it is unopened with the seal intact, so an opened item doesn't meet that condition for a change-of-mind return."
        elif category in {"grocery", "innerwear", "digital"}:
            answer = f"The catalog lists {product['title']} as {category}; it is {window}. Damaged, expired, wrong, or missing items follow a separate policy."
        else:
            answer = f"The catalog lists {product['title']} as {category}; the return window is {window}. The policy's condition rules also apply."
        if fee_question:
            plus_member = bool(re.search(r"\b(?:zentromart plus|plus member|i am plus|i'm plus|as a plus)\b", text))
            if category == "fashion" and plus_member:
                answer += " ZentroMart Plus members pay no reverse pickup fee."
            elif category == "fashion" and self._is_damage_report(message):
                answer += " No reverse pickup fee applies to damaged, defective or wrong-item returns."
            elif category == "fashion":
                answer += " A ₹49 reverse-pickup fee applies to change-of-mind returns; it is waived for Plus members and for damaged, defective or wrong items."
            else:
                answer += " The ₹49 reverse-pickup fee applies only to change-of-mind clothing, footwear and bag returns."
        return self._result(answer, "answer", sources)

    def _pincode_answer(self, row: dict[str, str]) -> dict[str, Any]:
        service = row.get("serviceable") == "True"
        cod = row.get("cod_available") == "True"
        express = row.get("express_available") == "True"
        answer = f"For pincode {row['pincode']} ({row['city']}), delivery is {'available' if service else 'not available'}"
        if service:
            answer += f"; cash on delivery is {'available' if cod else 'not available'} and Express is {'available' if express else 'not available'}"
        answer += "."
        return self._result(answer, "answer", [{"id": f"pincodes.csv:{row['pincode']}", "title": f"{row['city']} delivery record"}])

    @staticmethod
    def _asks_return_window(message: str) -> bool:
        text = message.lower()
        return (bool(re.search(r"\breturn(?:s|ed|ing)?\b|\bexchange\b|send (?:it )?back|वापस|लौट", text))
                and bool(re.search(r"\b(?:how long|how many|days?|window|when|within|deadline|eligible)\b", text)))

    @staticmethod
    def _return_category_from_text(message: str) -> str | None:
        text = message.lower()
        categories = (
            ("fashion", r"\b(?:shoes?|footwear|clothing|clothes|fashion|bags?)\b"),
            ("electronics", r"\b(?:electronics?|phones?|laptops?|accessories)\b"),
            ("home_kitchen", r"\b(?:furniture|home and kitchen|kitchen items?)\b"),
            ("beauty_sealed", r"\b(?:beauty|personal care|skincare|makeup)\b"),
            ("books_media", r"\b(?:books?|stationery)\b"),
            ("grocery", r"\b(?:groceries|grocery|fresh produce|perishables?)\b"),
            ("innerwear", r"\b(?:innerwear|lingerie|swimwear)\b"),
            ("digital", r"\b(?:gift cards?|digital codes?)\b"),
        )
        return next((category for category, pattern in categories if re.search(pattern, text)), None)

    def _general_return_window_answer(self, category: str) -> dict[str, Any]:
        answers = {
            "fashion": ("Clothing, footwear and bags have a 15-calendar-day return window from delivery. "
                        "Orders placed 1–15 October 2026 have 30 days; item condition rules and the applicable pickup fee also apply.",
                        ("RET-2", "RET-4", "RET-6", "RET-8")),
            "electronics": ("Electronics have a 7-calendar-day return window from delivery. Items delivered before 1 August 2026 keep the earlier 10-day window; condition rules apply.",
                            ("RET-2", "RET-3", "RET-4")),
            "home_kitchen": ("Home and kitchen items have a 10-calendar-day return window from delivery, subject to the policy's condition rules.",
                             ("RET-2", "RET-4")),
            "beauty_sealed": ("Beauty and personal-care items have a 7-calendar-day return window from delivery only if unopened with the seal intact.",
                              ("RET-2", "RET-4")),
            "books_media": ("Books and stationery have a 10-calendar-day return window from delivery, subject to the policy's condition rules.",
                            ("RET-2", "RET-4")),
            "grocery": ("Grocery and fresh items are not returnable for change of mind. Damaged, expired, wrong or missing items follow the separate reporting policy.",
                        ("RET-2", "RET-5", "DWM-1")),
            "innerwear": ("Innerwear and swimwear are not returnable for change of mind. Damaged, wrong or missing items follow the separate reporting policy.",
                          ("RET-2", "RET-5", "DWM-1")),
            "digital": ("Gift cards and digital codes are not returnable for change of mind. Contact support if a code is faulty.",
                        ("RET-2", "RET-5")),
        }
        answer, section_ids = answers[category]
        return self._result(answer, "answer", self._sources_by_id(*section_ids))

    def _refund_timeline(self, message: str, session: dict[str, Any]) -> dict[str, Any] | None:
        text = message.lower()
        temporal = any(term in text for term in (
            "when", "how long", "how fast", "how soon", "take", "arrive", "come",
            "kitne din", "kab tak", "by which date", "by what date", "date should"))
        if not temporal or not any(term in text for term in ("refund", "money back", "paisa")):
            # Continue an earlier clarification if the customer answers with a method or date.
            if session.get("refund_waiting_for") and (self._refund_method(text) or self._parse_start_date(text)):
                temporal = True
            elif self._refund_method(text) == "cod_unspecified" and any(term in text for term in ("refund", "money back", "paisa")) and "how" in text:
                temporal = True
            else:
                if session.get("refund_waiting_for"):
                    session["refund_waiting_for"] = None
                    session["refund_method"] = None
                    session["refund_start_date"] = None
                return None
        waiting_for = session.get("refund_waiting_for")
        method = self._refund_method(text)
        if not method and waiting_for in ("date", "cod_choice"):
            method = session.get("refund_method")
        start = self._parse_start_date(message)
        if start is None and waiting_for in ("method", "date") and session.get("refund_start_date"):
            try:
                start = date.fromisoformat(session["refund_start_date"])
            except ValueError:
                session["refund_start_date"] = None
        if start is not None and waiting_for in ("method", "date"):
            session["refund_start_date"] = start.isoformat()
        if method == "cod_unspecified":
            if not any(term in text for term in ("when", "how long", "how fast", "take", "arrive", "come", "kab tak", "kitne din")):
                session["refund_waiting_for"] = None
                session["refund_method"] = None
                session["refund_start_date"] = None
                return self._result("For a cash-on-delivery refund, you can choose ZentroMart Wallet (instant, within 2 hours) or a bank transfer (5 to 7 business days after bank details are verified).",
                                    "answer", self._sources_by_id("REF-3", "REF-2"))
            session["refund_waiting_for"] = "cod_choice"
            return self._result("For a cash-on-delivery refund, would you prefer instant Wallet credit or a bank transfer?", "clarify",
                                self._sources_by_id("REF-3"))
        if not method:
            session["refund_waiting_for"] = "method"
            session["refund_method"] = None
            if start is not None:
                session["refund_start_date"] = start.isoformat()
            return self._result("Which payment method was used: UPI, card, net banking, Wallet, or a cash-on-delivery refund?", "clarify",
                                self._sources_by_id("REF-2"))
        session["refund_method"] = method if waiting_for in ("method", "date", "cod_choice") else None
        if method == "wallet":
            session["refund_waiting_for"] = None
            session["refund_method"] = None
            session["refund_start_date"] = None
            return self._result("ZentroMart Wallet refunds are instant, within 2 hours of initiation.", "answer",
                                self._source_list(self._retrieve("ZentroMart Wallet refund instant within 2 hours")))
        if method == "cod_wallet":
            session["refund_waiting_for"] = None
            session["refund_method"] = None
            session["refund_start_date"] = None
            return self._result("A cash-on-delivery refund to ZentroMart Wallet is instant, within 2 hours of initiation.", "answer",
                                self._source_list(self._retrieve("cash on delivery Wallet refund instant")))
        if not start:
            if session.get("refund_waiting_for") == "date":
                session["refund_waiting_for"] = "date"
                return self._result("On what date was the refund initiated?", "clarify", self._sources_by_id("REF-2", "KB-00.3"))
            timeline = self._method_timeline(method)
            if session.get("refund_waiting_for") == "method":
                session["refund_waiting_for"] = "date"
                session["refund_method"] = method
                return self._result(f"{timeline} On what date was the refund initiated?", "clarify",
                                    self._sources_by_id("REF-2", "KB-00.3"))
            session["refund_waiting_for"] = None
            session["refund_method"] = None
            session["refund_start_date"] = None
            return self._result(timeline, "answer", self._source_list(self._retrieve("refund timeline payment method")))
        if start > TODAY:
            session["refund_waiting_for"] = "date"
            session["refund_method"] = method
            session["refund_start_date"] = start.isoformat()
            return self._result(
                "That initiation date is after the dataset's fixed date of 20 October 2026. Please check the date and send it again.",
                "clarify", self._sources_by_id("KB-00.2"))
        session["refund_waiting_for"] = None
        session["refund_method"] = None
        session["refund_start_date"] = None
        low, high = self._method_days(method)
        low_date = self._business_days_after(start, low)
        high_date = self._business_days_after(start, high)
        if low_date == high_date:
            answer = f"Based on the fixed dataset calendar, the refund should arrive by {self._format_date(low_date.isoformat())} ({low_date.isoformat()})."
        else:
            answer = f"Based on the fixed dataset calendar, the refund window is {self._format_date(low_date.isoformat())} to {self._format_date(high_date.isoformat())}; the latest date is {high_date.isoformat()}."
        return self._result(answer, "answer", self._sources_by_id("REF-2", "KB-00.3"))

    @staticmethod
    def _refund_method(text: str) -> str | None:
        lowered = text.lower()
        if "cash on delivery" in lowered or re.search(r"\bcod\b", lowered):
            if "wallet" in lowered:
                return "cod_wallet"
            if any(term in lowered for term in ("bank", "transfer", "neft")):
                return "cod_bank"
            return "cod_unspecified"
        if "wallet" in lowered:
            return "wallet"
        if "upi" in lowered:
            return "upi"
        if "net banking" in lowered or "netbanking" in lowered:
            return "net_banking"
        if "credit card" in lowered or "debit card" in lowered or re.search(r"\bcard\b", lowered):
            return "card"
        return None

    @staticmethod
    def _method_days(method: str) -> tuple[int, int]:
        return {"upi": (1, 3), "card": (5, 7), "net_banking": (3, 5), "cod_bank": (5, 7)}[method]

    def _method_timeline(self, method: str) -> str:
        low, high = self._method_days(method)
        label = {"upi": "UPI", "card": "card", "net_banking": "net banking", "cod_bank": "cash-on-delivery bank transfers"}[method]
        return f"{label.capitalize()} refunds take {low} to {high} business days after initiation."

    def _parse_start_date(self, text: str) -> date | None:
        iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if iso:
            try:
                return date.fromisoformat(iso.group(1))
            except ValueError:
                return None
        match = DATE_RE.search(text)
        if not match:
            return None
        month_token = match.group(2).lower()
        month = next((number for name, number in MONTHS.items() if name.startswith(month_token[:3])), None)
        if not month:
            return None
        try:
            return date(int(match.group(3) or TODAY.year), month, int(match.group(1)))
        except ValueError:
            return None

    @staticmethod
    def _business_days_after(start: date, count: int) -> date:
        if count == 0:
            return start
        day, elapsed = start, 0
        while elapsed < count:
            day += timedelta(days=1)
            if day.weekday() < 6 and day not in HOLIDAYS:
                elapsed += 1
        return day

    @staticmethod
    def _format_date(value: str | date | None) -> str | None:
        if not value:
            return None
        try:
            parsed = value if isinstance(value, date) else date.fromisoformat(str(value))
        except ValueError:
            return str(value)
        return f"{parsed.day} {parsed.strftime('%B')}"

    @staticmethod
    def _latest_timeline(timeline: str) -> str | None:
        parts = [part.strip() for part in timeline.split(";") if part.strip()]
        if not parts:
            return None
        part = parts[-1]
        match = re.match(r"(\d{4}-\d{2}-\d{2})T[^ ]+\s+(.*)", part)
        if match:
            formatted = SupportAgent._format_date(match.group(1))
            return f"{match.group(2).replace('_', ' ')} on {formatted}"
        return part.replace("_", " ")

    def _retrieve(self, query: str, limit: int = 3) -> list[Section]:
        retrieval_query = query.lower()
        # Common customer wording that differs from the policy vocabulary.
        phrase_aliases = {
            "money back": "refund", "get my money": "refund", "give my money": "refund",
            "send it back": "return", "give it back": "return", "take it back": "return",
            "pick it up": "pickup", "collect it": "pickup", "cash on delivery": "cod",
            "seller": "marketplace", "seller's": "marketplace", "bought from": "seller",
            "not arrived": "delivery", "where is my parcel": "track order",
            "last mile": "delivery", "exchange it": "return",
        }
        alias_topics: set[str] = set()
        for phrase, alias in phrase_aliases.items():
            if phrase in retrieval_query:
                retrieval_query += " " + alias
                alias_topics.add(alias)
        query_tokens = set(tokens(retrieval_query))
        query_tokens.update(alias_topics)
        lowered = query.lower()
        expansions = {
            "refund": {"refund", "timeline", "initiat", "payment"},
            "return": {"return", "window", "pickup", "fee"},
            "cancel": {"cancel", "ship", "order"},
            "stop": {"cancel", "ship", "shipped"},
            "warranty": {"warranty", "assured", "claim"},
            "assured": {"assured", "warranty", "claim"},
            "damag": {"damage", "damaged", "defective", "photo", "48"},
            "late": {"late", "express", "promised", "credit"},
        }
        for trigger, values in expansions.items():
            if trigger in lowered:
                query_tokens.update(values)
        focus_prefixes: tuple[str, ...] = ()
        if "refund" in query_tokens:
            focus_prefixes = ("REF-",)
        elif "return" in query_tokens:
            focus_prefixes = ("RET-", "DWM-")
        elif "pickup" in query_tokens:
            focus_prefixes = ("RET-", "DWM-", "HELP-REV")
        if not query_tokens:
            return []
        n_docs = max(1, len(self.sections))
        scored: list[tuple[float, Section]] = []
        for section in self.sections:
            counts = Counter(tokens(section.searchable_text))
            length = sum(counts.values()) or 1
            score = 0.0
            matches = 0
            for term in query_tokens:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                matches += 1
                df = self.document_frequency.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / max(1, self.avg_doc_len)))
            if matches == 0:
                continue
            score += {"policy": 0.5, "agent_guide": 0.2, "help": 0.1,
                      "faq": 0.0, "meta": 0.15, "marketing": -0.6}.get(section.doc_type, 0.0)
            if focus_prefixes and section.section_id.startswith(focus_prefixes):
                score += 0.8
            scored.append((score, section))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < 0.45:
            return []
        best = scored[0][0]
        # Avoid displaying weakly related sections as evidence. A tighter
        # cutoff keeps the source chips meaningful while retaining close ties.
        relevant = [(score, section) for score, section in scored if score >= best * 0.58]
        focused = [(score, section) for score, section in relevant
                   if focus_prefixes and section.section_id.startswith(focus_prefixes)]
        if focused:
            relevant = focused
        # If a current policy is a plausible match, lower-precedence help/FAQ/
        # marketing copy must not outrank or contradict it. Keep date metadata
        # alongside policy evidence when it also matched the question.
        policies = [(score, section) for score, section in relevant
                    if section.doc_type == "policy" and score >= best * 0.62]
        if policies:
            relevant = policies
        return [section for _score, section in relevant[:limit]]

    @staticmethod
    def _source_list(sections: list[Section]) -> list[dict[str, str]]:
        seen: set[str] = set()
        output = []
        for section in sections:
            if section.section_id not in seen:
                seen.add(section.section_id)
                output.append({"id": section.section_id, "title": section.title,
                               "type": section.doc_type, "status": section.status,
                               "effective_from": section.effective_from})
        return output

    def _sources_by_id(self, *section_ids: str) -> list[dict[str, str]]:
        by_id = {section.section_id: section for section in self.sections}
        return self._source_list([by_id[section_id] for section_id in section_ids if section_id in by_id])

    def _excerpt(self, section: Section, query: str) -> str:
        query_words = set(tokens(query))
        lines = []
        for line in section.body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("|") and set(stripped) <= set("|-: "):
                continue
            if stripped.startswith("#"):
                continue
            lines.append(stripped)
        if not lines:
            return "I found a matching policy section, but it doesn't contain a short extractive summary."
        ranked = sorted(enumerate(lines), key=lambda item: (
            len(query_words & set(tokens(item[1]))),
            not item[1].startswith("|"), -item[0]), reverse=True)
        chosen = [ranked[0][1]]
        # Tables need a header row when a matching data row was selected.
        if chosen[0].startswith("|"):
            header = next((line for line in lines if line.startswith("|") and not set(line) <= set("|-: ")), None)
            if header and header not in chosen:
                chosen.insert(0, header)
        if len(chosen[0]) < 100:
            neighbor_index = next((index for index, line in enumerate(lines) if line == chosen[-1]), -1)
            if 0 <= neighbor_index + 1 < len(lines) and not lines[neighbor_index + 1].startswith("|"):
                chosen.append(lines[neighbor_index + 1])
        excerpt = " ".join(chosen)
        excerpt = re.sub(r"\s+", " ", excerpt)
        if len(excerpt) > 380:
            excerpt = excerpt[:377].rsplit(" ", 1)[0] + "…"
        return excerpt


AGENT = SupportAgent()
PAGE = (SOURCE_ROOT / "index.html").read_bytes()
EMAIL_DRAFTS: dict[str, dict[str, Any]] = {}
EMAIL_DRAFTS_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "ZentroMartDemo/0.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        # Avoid putting customer text or identifiers in the terminal log.
        return

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, "application/json; charset=utf-8",
                   json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/" or path == "/index.html":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/api/health":
            self._json(200, {"ok": True,
                             "mode": f"{llm_adapter.provider_name()}_rag_available" if AGENT.llm_enabled else "offline",
                             "llm_enabled": AGENT.llm_enabled,
                             "llm_provider": llm_adapter.provider_name() if AGENT.llm_enabled else None,
                             "llm_model": llm_adapter.model_name() if AGENT.llm_enabled else None,
                             "policy_sections": len(AGENT.sections),
                             "email_mode": "local_mock_outbox",
                             "voice_enabled": voice_adapter.configured()})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in ("/api/chat", "/api/reset", "/api/email/send", "/api/voice"):
            self._json(404, {"error": "Not found"})
            return
        if self.headers.get_content_type().lower() != "application/json":
            self._json(415, {"error": "Content-Type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            max_length = 11_000_000 if path == "/api/voice" else 12000
            if length < 0 or length > max_length:
                self._json(413, {"error": "Request is too large"})
                return
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                self._json(400, {"error": "Request body must be a JSON object"})
                return
            session_id = data.get("session_id")
            if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", session_id):
                self._json(400, {"error": "A valid session_id is required"})
                return
            if path == "/api/reset":
                AGENT.reset(session_id)
                with EMAIL_DRAFTS_LOCK:
                    for draft_id in [key for key, draft in EMAIL_DRAFTS.items()
                                     if draft["session_id"] == session_id and draft["status"] == "pending"]:
                        EMAIL_DRAFTS.pop(draft_id, None)
                self._json(200, {"ok": True})
                return
            if path == "/api/voice":
                audio_text = data.get("audio_wav_base64")
                if not isinstance(audio_text, str) or len(audio_text) > 10_700_000:
                    self._json(400, {"error": "A valid short WAV voice clip is required."})
                    return
                try:
                    audio = base64.b64decode(audio_text, validate=True)
                    transcript = voice_adapter.transcribe_wav(audio)
                except voice_adapter.VoiceUnavailable:
                    self._json(503, {"error": "Gnani voice is not configured or the optional package is missing."})
                    return
                except (ValueError, binascii.Error):
                    self._json(400, {"error": "The voice clip is invalid. Record a short voice message and try again."})
                    return
                result = AGENT.respond(session_id, transcript)
                try:
                    audio_reply = voice_adapter.synthesize_stream(result["answer"])
                    audio_reply_b64 = base64.b64encode(audio_reply).decode("ascii")
                    speech_error = None
                except Exception:
                    audio_reply_b64 = None
                    speech_error = "The text reply is ready, but speech playback could not be generated."
                self._json(200, {"ok": True, "transcript": transcript,
                                 "answer": result["answer"], "decision": result["decision"],
                                 "sources": result.get("sources", []), "tool_calls": result.get("tool_calls", []),
                                 "engine": result.get("engine"), "model": result.get("model"),
                                 "audio_reply_base64": audio_reply_b64, "speech_error": speech_error})
                return
            if path == "/api/email/send":
                draft_id = data.get("draft_id")
                if not isinstance(draft_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{20,100}", draft_id):
                    self._json(400, {"error": "A valid email draft is required."})
                    return
                with EMAIL_DRAFTS_LOCK:
                    draft = EMAIL_DRAFTS.get(draft_id)
                    if not draft or draft["session_id"] != session_id:
                        self._json(404, {"error": "Email draft not found. Please prepare the reply again."})
                        return
                    if time.time() - draft["created_at"] > 3600:
                        EMAIL_DRAFTS.pop(draft_id, None)
                        self._json(410, {"error": "Email draft expired. Please prepare the reply again."})
                        return
                    if draft["status"] == "queued":
                        self._json(200, {"ok": True, "status": "already_queued"})
                        return
                    if draft["status"] == "sending":
                        self._json(409, {"error": "This email is already being sent."})
                        return
                    draft["status"] = "sending"
                try:
                    email_adapter.queue_email(draft["recipient"], draft["subject"], draft["body"])
                except Exception:
                    with EMAIL_DRAFTS_LOCK:
                        draft["status"] = "pending"
                    self._json(400, {"error": "The local email draft could not be saved. Check the address and try again."})
                    return
                with EMAIL_DRAFTS_LOCK:
                    draft["status"] = "queued"
                    draft["queued_at"] = time.time()
                self._json(200, {"ok": True, "status": "queued",
                                 "message": "Saved to the local demo outbox. No external email was sent."})
                return
            message = data.get("message")
            if not isinstance(message, str):
                self._json(400, {"error": "message must be text"})
                return
            channel = data.get("channel", "chat")
            if channel not in ("chat", "email", "ticket"):
                self._json(400, {"error": "channel must be chat, email, or ticket"})
                return
            result = AGENT.respond(session_id, message)
            result["channel"] = channel
            recipient = data.get("email_to")
            if channel == "email" and recipient is not None:
                try:
                    recipient = email_adapter.validate_email_address(recipient)
                except email_adapter.EmailValidationError as error:
                    self._json(400, {"error": str(error)})
                    return
                with AGENT.lock:
                    state = AGENT.sessions.get(session_id, {})
                    has_order_context = bool(state.get("pending_order_id") or state.get("verified_order"))
                if has_order_context:
                    result["email_status"] = "blocked_sensitive_order_context"
                    result["email_message"] = "For privacy, this demo cannot email order-specific details. Use the verified on-screen flow or request a human handover."
                else:
                    incoming_subject = data.get("email_subject", "")
                    if not isinstance(incoming_subject, str) or len(incoming_subject) > 180 or "\r" in incoming_subject or "\n" in incoming_subject:
                        self._json(400, {"error": "Email subject is invalid."})
                        return
                    subject = ("Re: " + incoming_subject.strip()) if incoming_subject.strip() else "ZentroMart customer support reply"
                    draft_id = secrets.token_urlsafe(24)
                    with EMAIL_DRAFTS_LOCK:
                        expired = [key for key, draft in EMAIL_DRAFTS.items()
                                   if time.time() - draft["created_at"] > 3600]
                        for key in expired:
                            EMAIL_DRAFTS.pop(key, None)
                        EMAIL_DRAFTS[draft_id] = {
                            "session_id": session_id, "recipient": recipient, "subject": subject,
                            "body": result["answer"], "created_at": time.time(), "status": "pending",
                        }
                    result["email_status"] = "draft_ready"
                    result["email_draft_id"] = draft_id
            self._json(200, result)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._json(400, {"error": "Invalid JSON request"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ZentroMart support-agent demo.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ZentroMart demo ready at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ZentroMart demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
