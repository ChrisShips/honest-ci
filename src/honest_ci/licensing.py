"""License keys: what one says, and whether honest-ci issued it.

Under the Fair Core License the license key check is the line between free and
paid, and it runs in a team's own CI, offline. So a key carries everything the
check needs to know, signed: which organization or account it is for, and
until when. The check verifies the signature against a public key it ships
with, and calls nobody.

A key is ``hci1.<payload>.<signature>``: the payload is UTF-8 JSON, both parts
are base64url without padding, and the signature is RSASSA-PKCS1-v1_5 with
SHA-256 over the ASCII of ``hci1.<payload>``, prefix included, so a key can't
be passed off as one of another format. That is the signature ``openssl dgst
-sha256 -sign`` makes, which is how keys are issued: the private key stays on
one computer, offline, and a key needs nothing else to be made.

Everything here reads its arguments only, the day included, and a refusal says
which part of a key is wrong without repeating the key: it is a credential,
and a refusal ends up in a pull request comment and a log.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

#: The one format this release reads. A payload names its version too
#: (``v``), so a later format can keep the prefix and still be told apart.
KEY_PREFIX = "hci1"
KEY_VERSION = 1

#: How long a key keeps working past its expiry date, as the terms of service
#: promise: a renewal that comes late must not stop a paying team's CI on the
#: day its period ends. A trial key has none.
GRACE_DAYS = 30

#: The plan of a trial key, the one key whose expiry date is its last day: it
#: stands in for the trial GitHub Actions counts on its own, which has no grace
#: days either, and nobody renews it.
TRIAL_PLAN = "trial"

#: DER of the DigestInfo for SHA-256 without its digest, as RFC 8017 §9.2
#: note 1 gives it: the bytes EMSA-PKCS1-v1_5 puts in front of the hash.
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")

_BASE64URL = re.compile(r"[A-Za-z0-9_-]+")
#: ASCII digits only: ``\d`` matches the digits of every script.
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


class LicenseKeyError(ValueError):
    """A key that doesn't let the gates run, and why, in words a pull request can show.

    The message names the part of the key that is wrong and never the key.
    """


@dataclass(frozen=True)
class PublicKey:
    """An RSA public key: the modulus ``n`` and the public exponent ``e``."""

    n: int
    e: int


#: The public key installations verify keys against: RSA-2048, made with
#: ``openssl genrsa 2048``. Its private half is kept offline, and never enters
#: this repository, CI or the sandbox, so a key opens only if honest-ci issued it.
PUBLIC_KEY = PublicKey(
    n=int(
        "229836409852314745151333562482557867281037988450882384089820584488216776"
        "764024797053565117239752004742844562101724329226207743734381405263859000"
        "150832878483765014899028801067116260441295649249629025109118208275183412"
        "968117574519325781801809090749011903282568327348570214551081732876737583"
        "294748095748635963523376463872035794828595922756023120249691745291386202"
        "427937221532826095843806188048060748605955757975926095888490810368327253"
        "660464579669263855479449611504027843432622510537387206864352942931000677"
        "322175886482147327209303998445397777926218724326053395635624127983141703"
        "65885956527878959730101169319108241226443"
    ),
    e=65537,
)


@dataclass(frozen=True)
class LicenseKey:
    """What a key honest-ci issued says. Nothing here is the key itself."""

    #: The license's number, so a customer's email and a key can be matched up.
    id: str
    #: The numeric id of the organization or personal account the key is for,
    #: compared with the id of the repository's owner. The id, not the name:
    #: a renamed organization keeps its key.
    org_id: int
    #: That organization's or account's login when the key was issued. Shown,
    #: never compared.
    org: str
    #: The plan, such as ``small``, ``team``, ``beta``, ``trial`` or
    #: ``personal``. Shown, and compared only to tell a trial key, which has
    #: no grace days: every plan runs the same honest-ci.
    plan: str
    #: How many people the subscription is for. Shown, never counted.
    seats: int
    issued: date
    #: The last day the subscription paid for, or the trial's last day, in UTC.
    expires: date

    @property
    def is_trial(self) -> bool:
        """Whether this is a trial key, whose expiry date is its last day."""

        return self.plan == TRIAL_PLAN

    @property
    def works_until(self) -> date:
        """The last day this key lets the gates run: its expiry date, then any grace days."""

        return self.expires if self.is_trial else self.expires + timedelta(days=GRACE_DAYS)

    def in_grace(self, today: date) -> bool:
        """Whether ``today`` falls in the days after the expiry date that the key still works."""

        return self.expires < today <= self.works_until


def read_key(text: str, *, public_key: PublicKey) -> LicenseKey:
    """The key ``text``, when ``public_key`` verifies it.

    Raises ``LicenseKeyError`` when it isn't a key in this format, its signature
    doesn't verify, or what it signs isn't a payload this release reads. Who
    the key is for and whether it is still current are ``check_key``'s.
    """

    # A key pasted into a secret often brings the line break after it. No key
    # holds whitespace, so taking it off the ends can't make a wrong key right.
    parts = text.strip().split(".")
    if len(parts) != 3 or parts[0] != KEY_PREFIX:
        raise LicenseKeyError(
            f"it isn't an honest-ci license key, which is {KEY_PREFIX}.<payload>.<signature> "
            "on one line"
        )
    _, payload, signature = parts
    try:
        payload_bytes = _base64url(payload)
        signature_bytes = _base64url(signature)
    except ValueError:
        raise LicenseKeyError(
            "it isn't an honest-ci license key: its payload or signature isn't base64url "
            "without padding, as when a key is copied with a line break in it or cut short"
        ) from None
    if not verify_signature(
        f"{KEY_PREFIX}.{payload}".encode("ascii"), signature_bytes, public_key
    ):
        raise LicenseKeyError(
            "its signature doesn't match: the key is incomplete or was changed after it "
            "was issued, or honest-ci didn't issue it"
        )
    try:
        fields = json.loads(payload_bytes.decode("utf-8"))
    except ValueError:
        raise LicenseKeyError("its payload isn't JSON") from None
    if not isinstance(fields, dict):
        raise LicenseKeyError("its payload isn't a JSON object")
    # Checked before the other fields: a later format is free to change them.
    if "v" not in fields:
        raise LicenseKeyError("its payload has no version")
    version = fields["v"]
    # ``type`` rather than ``isinstance``: JSON's ``true`` is a ``bool``, which
    # Python counts as an ``int`` equal to 1, and ``1.0`` equals 1 too.
    if type(version) is not int or version != KEY_VERSION:
        raise LicenseKeyError(
            f"it is a version {_shown(version)} key, and this release of honest-ci reads "
            f"version {KEY_VERSION}; a newer release may read it"
        )
    return LicenseKey(
        id=_text(fields, "id"),
        org_id=_count(fields, "org_id", least=1),
        org=_text(fields, "org"),
        plan=_text(fields, "plan"),
        seats=_count(fields, "seats", least=0),
        issued=_day(fields, "issued"),
        expires=_day(fields, "expires"),
    )


def check_key(text: str, *, owner_id: int, today: date, public_key: PublicKey) -> LicenseKey:
    """The key ``text``, when it lets the gates run today on a repository ``owner_id`` owns.

    That takes everything ``read_key`` asks, and two things more: the key is
    for the repository's owner, and today is no later than ``works_until``.
    A key past its expiry date but in its grace days passes; the caller says
    so on the run (``LicenseKey.in_grace``). A trial key has none.
    """

    key = read_key(text, public_key=public_key)
    if key.org_id != owner_id:
        raise LicenseKeyError(
            f"it was issued for {key.org} (id {key.org_id}), and this repository's owner "
            f"has id {owner_id}"
        )
    refused = expiry_refusal(key, today)
    if refused is not None:
        raise LicenseKeyError(refused)
    return key


def expiry_refusal(key: LicenseKey, today: date) -> str | None:
    """Why ``key`` no longer lets the gates run on ``today``, or ``None`` while it does.

    In a refusal's words: ``check_key`` refuses a key with them, and ``doctor``
    shows them of the key it is given.
    """

    if today <= key.works_until:
        return None
    if key.is_trial:
        return f"it is a trial key that expired on {key.expires.isoformat()}"
    return f"it expired on {key.expires.isoformat()}, more than {GRACE_DAYS} days ago"


def verify_signature(message: bytes, signature: bytes, public_key: PublicKey) -> bool:
    """Whether ``signature`` is RSASSA-PKCS1-v1_5 with SHA-256 of ``message`` (RFC 8017 §8.2.2).

    The signature is opened, and the result compared whole, in constant time,
    with the one encoding RFC 8017 §9.2 gives for this message. Nothing in the
    opened block is parsed: verifiers that parsed the padding and the
    DigestInfo, and let slack through, are how forged signatures have passed.
    """

    size = (public_key.n.bit_length() + 7) // 8
    if len(signature) != size:
        return False
    representative = int.from_bytes(signature, "big")
    if representative >= public_key.n:
        # RFC 8017 §5.2.2: out of range. It would open as its remainder, which
        # makes a second spelling of a valid signature rather than a forgery,
        # but a key has one signature.
        return False
    opened = pow(representative, public_key.e, public_key.n).to_bytes(size, "big")
    return hmac.compare_digest(opened, _encoded(message, size))


def _encoded(message: bytes, size: int) -> bytes:
    """EMSA-PKCS1-v1_5-ENCODE of ``message`` with SHA-256, ``size`` bytes long (RFC 8017 §9.2)."""

    digest_info = _SHA256_DIGEST_INFO + hashlib.sha256(message).digest()
    padding = size - len(digest_info) - 3
    return b"\x00\x01" + b"\xff" * padding + b"\x00" + digest_info


def _base64url(text: str) -> bytes:
    """``text`` decoded as base64url without padding; ``ValueError`` for anything else.

    ``base64.urlsafe_b64decode`` alone would skip characters outside the
    alphabet, so a key with a stray character in it would decode, and be
    refused later for a reason that sends the reader to the wrong place.
    """

    if not _BASE64URL.fullmatch(text):
        raise ValueError("not base64url")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except binascii.Error as error:
        raise ValueError("not base64url") from error


def _text(fields: dict[str, Any], name: str) -> str:
    value = fields.get(name)
    if not isinstance(value, str) or not value.strip():
        raise LicenseKeyError(f"its payload has no {name}")
    return value


def _count(fields: dict[str, Any], name: str, *, least: int) -> int:
    value = fields.get(name)
    # ``type``, as for the version: ``true`` and ``3.0`` aren't counts.
    if type(value) is not int or value < least:
        raise LicenseKeyError(f"its payload has no {name}")
    return value


def _day(fields: dict[str, Any], name: str) -> date:
    value = fields.get(name)
    # The one spelling, YYYY-MM-DD: ``date.fromisoformat`` also takes week dates
    # and dates without hyphens, which no key is issued with.
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        raise LicenseKeyError(f"its payload has no {name} date")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise LicenseKeyError(f"its payload has no {name} date") from None


def _shown(value: object) -> str:
    """``value`` as a refusal quotes it, cut short: it is whatever the payload holds."""

    shown = json.dumps(value)
    return shown if len(shown) <= 20 else shown[:17] + "..."


__all__ = [
    "GRACE_DAYS",
    "KEY_PREFIX",
    "KEY_VERSION",
    "PUBLIC_KEY",
    "TRIAL_PLAN",
    "LicenseKey",
    "LicenseKeyError",
    "PublicKey",
    "check_key",
    "expiry_refusal",
    "read_key",
    "verify_signature",
]
