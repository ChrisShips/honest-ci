"""Provider-agnostic structured-JSON model adapter.

Every hosted model this product talks to obeys the same contract: redact the
payload, ask for exactly one JSON object, validate it against a Pydantic
artifact, retry within a bounded budget, and record model/prompt/token metadata.
Only the wire details differ (protocol, base URL, model id, credential, sampling
limits, provider-specific body fields), so those are described by
``ModelEndpoint`` and every provider subclass supplies one.

Concrete providers live in ``deepseek.py``, ``qwen.py`` (Alibaba Bailian),
``glm.py`` (Z.ai), ``openai.py`` (any OpenAI-compatible server) and
``anthropic.py``. Nothing here relaxes a gate: a schema-valid response is still
only a proposal that ``grounding.py`` must accept.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from ..contracts import (
    BlockBrief,
    ChangeRequest,
    EvidencePack,
    GateName,
    LLMCallRecord,
    ModelReviewReport,
    ProposedTests,
)
from ..redaction import redact_obj
from ..settings import MaxTokensParam, Settings
from ..utils import open_validated_http_request
from .base import LLMAdapter

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

#: Diff bytes sent for a review. Large enough for a real MR, bounded so one
#: giant refactor cannot blow the context window or the bill.
MAX_REVIEW_DIFF_CHARS = 60000

#: Changed symbols offered as generation targets in one call. A change that
#: touched more than this has a blast radius no generated case will cover.
MAX_TESTGEN_SYMBOLS = 40


class StructuredModelError(RuntimeError):
    """A provider call, its JSON extraction, or its schema validation failed."""


@dataclass(frozen=True)
class ModelEndpoint:
    """Everything that differs between providers for one structured call."""

    provider: str
    protocol: Literal["anthropic", "openai"]
    base_url: str
    model: str
    api_key: str | None
    max_tokens: int
    #: ``None`` sends no ``temperature`` at all: OpenAI's reasoning models and
    #: current Claude models reject the field rather than ignore it.
    temperature: float | None
    timeout_seconds: int
    max_retries: int
    extra_body: dict[str, Any] = field(default_factory=dict)
    #: The OpenAI-shaped body's name for ``max_tokens``. The Anthropic
    #: Messages API has only the one name, so it ignores this.
    max_tokens_param: MaxTokensParam = "max_tokens"


class StructuredModelAdapter(LLMAdapter):
    """Shared prompt/validation/telemetry layer for hosted JSON model providers."""

    error_cls: ClassVar[type[StructuredModelError]] = StructuredModelError
    missing_credential_message: ClassVar[str] = "An API key is required for this model adapter."

    def __init__(self, settings: Settings, transport: Any | None = None) -> None:
        self.settings = settings
        self.transport = transport
        self.call_records: list[LLMCallRecord] = []

    def endpoint(self) -> ModelEndpoint:
        raise NotImplementedError

    # ── judgment tasks ───────────────────────────────────────────────────

    def review_change(self, evidence: EvidencePack) -> ModelReviewReport:
        change = evidence.change
        return self._structured(
            ModelReviewReport,
            (
                "You are assisting a human reviewer on a merge request. Never approve or reject it. "
                "Only raise a finding you can ground in a line the diff ADDS; quote its file and line "
                "numbers from the diff itself. If you are unsure, raise nothing and put the doubt in "
                "review_questions instead. Never recommend bypassing rules, tests, scanners or human "
                "review. Prefer team-rule and architecture violations over style nits."
            ),
            {
                "evidence_fingerprint": evidence.fingerprint,
                "change": {
                    "title": change.title,
                    "description": change.description,
                    "repo": change.repo,
                    "number": change.number,
                    "source_branch": change.source_branch,
                    "target_branch": change.target_branch,
                    "files_changed": [item.model_dump(mode="json") for item in change.files_changed],
                    "diff": change.diff[:MAX_REVIEW_DIFF_CHARS],
                },
                "gate_reports": [
                    {
                        "gate": report.gate,
                        "verdict": report.verdict.value,
                        "summary": report.summary,
                        "findings": [
                            {
                                "id": finding.id,
                                "rule_id": finding.rule_id,
                                "severity": finding.severity,
                                "title": finding.title,
                                "file": finding.file,
                            }
                            for finding in report.findings
                        ],
                    }
                    for report in evidence.gate_reports
                ],
            },
        )

    def judge_rules(
        self,
        *,
        change: ChangeRequest,
        fingerprint: str,
        rules: list[dict[str, str]],
        locations: list[dict[str, str]],
        reviews: Sequence[dict[str, str]] = (),
    ) -> ModelReviewReport:
        return self._structured(
            ModelReviewReport,
            (
                "You are applying one team's own review rules to a merge request. Judge ONLY "
                "against the rules in `rules`, and raise a finding only when a line in "
                "`locations` violates one of them. Set rule_id to that rule's id -- never "
                "invent one -- and copy file and line numbers from `locations` verbatim. "
                "Set confidence to how sure you are that this specific line violates that "
                "specific rule; a low number is a useful answer and an inflated one is not. "
                "When you are unsure, raise nothing and put the doubt in review_questions "
                "instead: handing an unsure case to a person is the designed outcome, not a "
                "failure. Do not report "
                "style, naming or anything no supplied rule covers. Severity comes from the "
                "rule file, so do not try to set it. `reviews` holds comments people (or the "
                "platform's own assistant) already left on this change: read them for context "
                "about what has been questioned already, and never cite one as justification -- "
                "a finding must name a rule id from `rules` and a line from `locations`, and a "
                "point somebody already made on the merge request does not need making again. "
                "Never approve or reject the change, and "
                "never recommend bypassing rules, tests, scanners or human review."
            ),
            {
                "evidence_fingerprint": fingerprint,
                "change": {
                    "title": change.title,
                    "description": change.description,
                    "repo": change.repo,
                    "source_branch": change.source_branch,
                    "target_branch": change.target_branch,
                    "diff": change.diff[:MAX_REVIEW_DIFF_CHARS],
                },
                "rules": rules,
                "locations": locations,
                "reviews": list(reviews),
            },
        )

    def propose_tests(
        self,
        *,
        change: ChangeRequest,
        symbols: Sequence[dict[str, str]],
        fingerprint: str,
        language: str,
        test_directory: str,
    ) -> ProposedTests:
        return self._structured(
            ProposedTests,
            (
                "Write regression tests for the symbols this change touched. A regression test "
                "pins down behaviour that exists NOW so a later change cannot break it silently, "
                "so assert on what the code in the diff actually does -- never on what you think "
                "it should do. Every test file's path must start with the given test_directory "
                "and must not be a file that already exists; list in `targets` only symbol names "
                "taken verbatim from `symbols`, and propose nothing for a symbol you cannot write "
                "a meaningful test for: fewer honest cases are worth more than one case per "
                "symbol. Each file must be complete and runnable on its own, import only the "
                "standard library, the repository's own modules and the test framework already in "
                "use, and must not reach the network, the clock, a database or any path outside a "
                "temporary directory -- the sandbox has no network and a flaky case is worse than "
                "no case. Do not modify, delete or re-create any existing file, and do not touch "
                "configuration, CI or build files. Never write a test that asserts nothing, that "
                "skips itself, or that only checks a mock you set up in the same test."
            ),
            {
                "evidence_fingerprint": fingerprint,
                "language": language,
                "test_directory": test_directory,
                "change": {
                    "title": change.title,
                    "description": change.description,
                    "repo": change.repo,
                    "diff": change.diff[:MAX_REVIEW_DIFF_CHARS],
                },
                "symbols": list(symbols)[:MAX_TESTGEN_SYMBOLS],
            },
        )

    def explain_block(
        self,
        *,
        gate: GateName,
        attempt: int,
        source_fingerprint: str,
        summary: str,
        deterministic_feedback: str,
        evidence_items: list[dict[str, str]],
    ) -> BlockBrief:
        return self._structured(
            BlockBrief,
            (
                "Explain why this gate blocked the change, for the developer who has to fix it. "
                "The gate is authoritative: never claim it passed, never recommend bypassing rules, "
                "tests, scanners or review. Cite only ids from evidence_items. Keep recommended "
                "actions minimal and concrete."
            ),
            {
                "gate": gate,
                "attempt": attempt,
                "source_fingerprint": source_fingerprint,
                "gate_summary": summary,
                "deterministic_feedback": deterministic_feedback,
                "evidence_items": evidence_items,
            },
        )

    # ── call machinery ───────────────────────────────────────────────────

    def _structured(self, schema: type[T], instruction: str, payload: dict[str, Any]) -> T:
        def validate(raw: dict[str, Any]) -> T:
            try:
                return schema.model_validate(raw)
            except ValidationError as exc:
                raise self.error_cls(
                    f"Model output failed {schema.__name__} validation: {exc}"
                ) from exc

        schema_payload = {
            **payload,
            # The real Pydantic JSON Schema, so arrays, enums and numeric
            # bounds are not left to provider interpretation.
            "schema": schema.model_json_schema(),
        }
        return self._call_json(
            instruction + " Follow the supplied JSON Schema exactly, including field types and enums.",
            schema_payload,
            schema_name=schema.__name__,
            validator=validate,
        )

    def _call_json(
        self,
        instruction: str,
        payload: dict[str, Any],
        *,
        schema_name: str,
        validator: Callable[[dict[str, Any]], R],
    ) -> R:
        endpoint = self.endpoint()
        if not endpoint.api_key:
            raise self.error_cls(self.missing_credential_message)
        safe_payload = redact_obj(payload)
        prompt = "\n".join(
            [
                instruction,
                "Return exactly one JSON object and no markdown.",
                json.dumps(safe_payload, ensure_ascii=False, indent=2, default=str),
            ]
        )
        attempts = 0
        last_error: str | None = None
        while attempts <= endpoint.max_retries:
            attempts += 1
            start = time.perf_counter()
            try:
                text, usage = self._send(prompt)
                data = extract_json(text, self.error_cls)
                result = validator(data)
                self.call_records.append(
                    LLMCallRecord(
                        provider=endpoint.provider,
                        protocol=endpoint.protocol,
                        model=endpoint.model,
                        prompt_version=self.settings.prompt_version,
                        latency_s=round(time.perf_counter() - start, 3),
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        attempts=attempts,
                        schema_name=schema_name,
                        schema_valid=True,
                    )
                )
                return result
            except Exception as exc:  # noqa: BLE001 - record and retry bounded model calls.
                last_error = str(exc)
                if attempts > endpoint.max_retries:
                    self.call_records.append(
                        LLMCallRecord(
                            provider=endpoint.provider,
                            protocol=endpoint.protocol,
                            model=endpoint.model,
                            prompt_version=self.settings.prompt_version,
                            latency_s=round(time.perf_counter() - start, 3),
                            attempts=attempts,
                            schema_name=schema_name,
                            schema_valid=False,
                            error=last_error,
                        )
                    )
                    raise
        raise self.error_cls(last_error or "Model call failed.")

    def _send(self, prompt: str) -> tuple[str, dict[str, int | None]]:
        if self.transport is not None:
            result: tuple[str, dict[str, int | None]] = self.transport(prompt, self.settings)
            return result
        if self.endpoint().protocol == "openai":
            return self._send_openai(prompt)
        return self._send_anthropic(prompt)

    def _send_openai(self, prompt: str) -> tuple[str, dict[str, int | None]]:
        endpoint = self.endpoint()
        url = f"{endpoint.base_url}/chat/completions"
        body = self._body(endpoint, prompt, max_tokens_param=endpoint.max_tokens_param)
        payload = self._request(url, body, {"Authorization": f"Bearer {endpoint.api_key}"})
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        return str(text), {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    def _send_anthropic(self, prompt: str) -> tuple[str, dict[str, int | None]]:
        endpoint = self.endpoint()
        url = f"{endpoint.base_url}/v1/messages"
        body = self._body(endpoint, prompt, max_tokens_param="max_tokens")
        payload = self._request(
            url,
            body,
            {"x-api-key": endpoint.api_key or "", "anthropic-version": "2023-06-01"},
        )
        chunks = payload.get("content", [])
        text = "\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text")
        usage = payload.get("usage", {})
        total = (
            usage.get("input_tokens", 0) + usage.get("output_tokens", 0) if usage else None
        )
        return text, {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": total,
        }

    @staticmethod
    def _body(
        endpoint: ModelEndpoint,
        prompt: str,
        *,
        max_tokens_param: MaxTokensParam,
    ) -> dict[str, Any]:
        """The request body both wire shapes share, minus what was left unset."""

        body: dict[str, Any] = {
            "model": endpoint.model,
            max_tokens_param: endpoint.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if endpoint.temperature is not None:
            body["temperature"] = endpoint.temperature
        return {**body, **endpoint.extra_body}

    def _request(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        endpoint = self.endpoint()
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "honest-ci",
                **headers,
            },
        )
        try:
            # The initial URL and every redirect remain under the configured
            # provider base, so a redirect cannot walk the credential elsewhere.
            with open_validated_http_request(
                request,
                allowed_base=endpoint.base_url,
                timeout_seconds=endpoint.timeout_seconds,
                allow_loopback_http=True,
            ) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise self.error_cls(f"{endpoint.provider} API failed: {exc.code} {detail}") from exc


def extract_json(
    text: str,
    error_cls: type[StructuredModelError] = StructuredModelError,
) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise error_cls("Model output did not contain a JSON object.")
    result: dict[str, Any] = json.loads(stripped[start : end + 1])
    return result
