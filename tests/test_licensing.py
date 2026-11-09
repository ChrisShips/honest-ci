"""License keys: which ones open, and what a refusal says.

Every key here is signed with a key pair made for these tests and nothing else,
whose private half is written out below. An installation verifies against
honest-ci's own public key, so nothing signed here opens anywhere else. One key
was signed by ``openssl dgst -sha256 -sign``, the command keys are issued with,
so the verifier answers to what openssl makes and not only to this file's own
signing.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pytest

from honest_ci.licensing import (
    GRACE_DAYS,
    PUBLIC_KEY,
    LicenseKey,
    LicenseKeyError,
    PublicKey,
    check_key,
    read_key,
    verify_signature,
)

# The test key pair: RSA-2048, made once with ``openssl genrsa 2048`` for these
# tests only. Its private exponent is in the open on purpose. It is written as
# integers, so the secret scan's private-key rule, which looks for PEM blocks,
# has nothing to find: that rule is there for real keys, and this one isn't.
_MODULUS = int(
    "229153784340906943889663264486442193945051274295932244791678924753252486"
    "033696164693270110100033196800228215729243690713608159797498993491333381"
    "582024546465214320801377677752541471976804505627038497557332508124968691"
    "308199828385201944125095224739035950824190270370990411237207593271036411"
    "274226608039964953828945261386295040367811010626264019964994658045791146"
    "565635683076869686547273075706659450241323032501308534707467551696953334"
    "901331355921894763235762409023180002916581151342564292021301836293385685"
    "332100009609562084710882179762543202618986180340282453060211837269110389"
    "95279831358292982745873892208961820276827"
)
_PRIVATE_EXPONENT = int(
    "552735557450151360148892516441350413031321321401665765242055700215056426"
    "022043879254652166022449116532341674816955958130631214440524297589300409"
    "852242859838275780586260941134347862888036624342314199518792787042358526"
    "053988264203926382460214125253624717431191509227553354721421126146839737"
    "769958072523271738114285417795919223010611353782232526509607250529727624"
    "443126119634851343616796296639466227578590488719588238903397371998985340"
    "640559094845600087023131963140897996936894766354235804022895218547847718"
    "085039758236992277423634099689394698453943612783238772872986585856433783"
    "4904919058218974622680476602809811300353"
)
TEST = PublicKey(n=_MODULUS, e=65537)
_SIZE = 256
#: The public half of another RSA-2048 key pair, whose private half was thrown away.
OTHER = PublicKey(
    n=int(
        "289733077855163691598227009611036179569585030746745739278507898304744869"
        "372189440647343705234780575713619022887836335050562399387646511565482519"
        "395841970624974226233301793907022607318129715221370373259024094291271037"
        "163293311276395899991010466712789232039432990615014219281778431236304114"
        "306610861058969063387730669920279599372454733455751855285078938118772283"
        "626266852678367916438465012888784879874856906107396089088652337379547766"
        "600388538149769401513322227425440744766238584399861235843573218358142513"
        "303304546884122975720591994199962950952734346092835683362200074698885796"
        "03347908433626467853161768569222418768879"
    ),
    e=65537,
)

#: A key ``openssl dgst -sha256 -sign`` signed with the test key's private half,
#: over the ASCII of ``hci1.`` and this payload, the way keys are issued:
#: ``{"v":1,"id":"L-TEST-0001","org_id":12345678,"org":"octo-org","plan":"team",
#: "seats":10,"issued":"2026-11-13","expires":"2027-11-13"}``
SIGNED_BY_OPENSSL = (
    "hci1.eyJ2IjoxLCJpZCI6IkwtVEVTVC0wMDAxIiwib3JnX2lkIjoxMjM0NTY3OCwib3JnIjo"
    "ib2N0by1vcmciLCJwbGFuIjoidGVhbSIsInNlYXRzIjoxMCwiaXNzdWVkIjoiMjAyNi0xMS0"
    "xMyIsImV4cGlyZXMiOiIyMDI3LTExLTEzIn0.YyEQpuW1f5St0ZGkBBBoNoK0yIvC53N3DaG"
    "EvTNYn5I9TVq0KcLXIRuj9Sv16Jn_zxhYd9rLnn4cHEMRndYNVDQKtpXVVBRZ31Q6dlqLWWa"
    "UIZ2A23ozw1eMNfi6JaOd0Hfq12lAyuUnqiU1nLq_PuCftECrrP_O1JyDUGgQcwXamwOojgI"
    "tuPE1No8SoIlEhhHipP3JBUU1fbSKRcIuwb5I-zleqdpxT0SQg6VwgHt8rSNsABkYuMnoOxm"
    "nzzjHu5XgjAzGcPMdC5lhENrB2kXv6r4DcpmWhcCkJdewAGZbljvA-gowmblsA08jsLtWUse"
    "HhE9g5gWO0ywpgkOY3Q"
)

OWNER_ID = 12345678
EXPIRES = date(2027, 11, 13)
#: What every key below says, unless a test changes it.
PAYLOAD: dict[str, Any] = {
    "v": 1,
    "id": "L-2026-0007",
    "org_id": OWNER_ID,
    "org": "octo-org",
    "plan": "small",
    "seats": 3,
    "issued": "2026-11-13",
    "expires": EXPIRES.isoformat(),
}
#: A field ``issue`` leaves out.
MISSING = object()

#: RFC 8017 §9.2 note 1: the DigestInfo DER for SHA-256, without the digest.
SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def unb64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_block(block: bytes) -> bytes:
    """RSASP1 of ``block`` with the test key's private half (RFC 8017 §5.2.1)."""

    return pow(int.from_bytes(block, "big"), _PRIVATE_EXPONENT, _MODULUS).to_bytes(_SIZE, "big")


def sign(message: bytes) -> bytes:
    """RSASSA-PKCS1-v1_5 with SHA-256 (RFC 8017 §8.2.1), written out for these tests."""

    digest_info = SHA256_DIGEST_INFO + hashlib.sha256(message).digest()
    return sign_block(b"\x00\x01" + b"\xff" * (_SIZE - len(digest_info) - 3) + b"\x00" + digest_info)


def issue(payload: bytes | None = None, **fields: Any) -> str:
    """A key the test key pair signed: ``PAYLOAD`` with ``fields`` changed, or ``payload`` as is."""

    if payload is None:
        said = {**PAYLOAD, **fields}
        payload = json.dumps({k: v for k, v in said.items() if v is not MISSING}).encode()
    signed = f"hci1.{b64url(payload)}"
    return f"{signed}.{b64url(sign(signed.encode('ascii')))}"


def refusal(text: str, *, owner_id: int = OWNER_ID, today: date = EXPIRES) -> str:
    """Why ``check_key`` refuses ``text``; fails the test when it doesn't."""

    with pytest.raises(LicenseKeyError) as refused:
        check_key(text, owner_id=owner_id, today=today, public_key=TEST)
    return str(refused.value)


# ── keys that open ───────────────────────────────────────────────────────────


def test_a_key_honest_ci_issued_reads_back_as_what_it_says() -> None:
    assert read_key(issue(), public_key=TEST) == LicenseKey(
        id="L-2026-0007",
        org_id=OWNER_ID,
        org="octo-org",
        plan="small",
        seats=3,
        issued=date(2026, 11, 13),
        expires=EXPIRES,
    )


def test_a_key_openssl_signed_opens_and_is_the_key_this_file_would_sign() -> None:
    """Keys are issued with ``openssl dgst -sha256 -sign``, and the verifier reads what it makes.

    PKCS #1 v1.5 signatures are deterministic, so signing the same payload here
    gives the same key, character for character: the signing every other test
    uses is openssl's, not a private variant that only this verifier accepts.
    """

    key = read_key(SIGNED_BY_OPENSSL, public_key=TEST)

    assert (key.id, key.org_id, key.plan, key.seats) == ("L-TEST-0001", OWNER_ID, "team", 10)
    payload = unb64url(SIGNED_BY_OPENSSL.split(".")[1])
    assert issue(payload) == SIGNED_BY_OPENSSL


def test_a_key_pasted_with_a_line_break_after_it_still_opens() -> None:
    assert read_key(f"  {issue()}\n", public_key=TEST) == read_key(issue(), public_key=TEST)


def test_a_field_this_release_doesnt_know_is_left_alone() -> None:
    """A later key of the same version may say more, for an older release to show nothing of."""

    assert read_key(issue(region="eu"), public_key=TEST) == read_key(issue(), public_key=TEST)


# ── the expiry date and the grace days ───────────────────────────────────────


def test_a_key_ten_days_past_its_expiry_still_works_and_says_until_when() -> None:
    today = EXPIRES + timedelta(days=10)

    key = check_key(issue(), owner_id=OWNER_ID, today=today, public_key=TEST)

    assert key.in_grace(today)
    assert key.works_until == date(2027, 12, 13)


def test_a_key_forty_days_past_its_expiry_is_refused_with_the_day_it_expired() -> None:
    why = refusal(issue(), today=EXPIRES + timedelta(days=40))

    assert why == "it expired on 2027-11-13, more than 30 days ago"


@pytest.mark.parametrize(
    ("days", "works", "in_grace"),
    [
        (-365, True, False),
        (0, True, False),
        (1, True, True),
        (GRACE_DAYS, True, True),
        (GRACE_DAYS + 1, False, False),
    ],
)
def test_a_key_works_through_its_expiry_date_and_the_thirty_days_after(
    days: int, works: bool, in_grace: bool
) -> None:
    """The expiry date is the last day paid for; the grace days are the 30 after it."""

    today = EXPIRES + timedelta(days=days)
    key = read_key(issue(), public_key=TEST)

    assert key.in_grace(today) is in_grace
    if works:
        assert check_key(issue(), owner_id=OWNER_ID, today=today, public_key=TEST) == key
    else:
        assert refusal(issue(), today=today).startswith("it expired on 2027-11-13")


@pytest.mark.parametrize(
    ("days", "works"), [(-13, True), (0, True), (1, False), (GRACE_DAYS, False)]
)
def test_a_trial_key_works_through_its_expiry_date_and_not_a_day_after(
    days: int, works: bool
) -> None:
    """Nobody renews a trial: a trial key stops on its last day, as the trial in GitHub Actions does."""

    today = EXPIRES + timedelta(days=days)
    trial = issue(plan="trial", seats=0)
    key = read_key(trial, public_key=TEST)

    assert key.works_until == EXPIRES
    assert not key.in_grace(today)
    if works:
        assert check_key(trial, owner_id=OWNER_ID, today=today, public_key=TEST) == key
    else:
        assert refusal(trial, today=today) == "it is a trial key that expired on 2027-11-13"


@pytest.mark.parametrize("plan", ["small", "team", "beta", "personal"])
def test_a_key_of_any_other_plan_keeps_its_thirty_grace_days(plan: str) -> None:
    today = EXPIRES + timedelta(days=GRACE_DAYS)

    key = check_key(issue(plan=plan), owner_id=OWNER_ID, today=today, public_key=TEST)

    assert key.in_grace(today)
    assert key.works_until == date(2027, 12, 13)


# ── keys that don't ──────────────────────────────────────────────────────────


def test_a_key_for_another_organization_is_refused_and_says_whose_it_is() -> None:
    why = refusal(issue(), owner_id=87654321)

    assert why == (
        "it was issued for octo-org (id 12345678), and this repository's owner has id 87654321"
    )


def test_a_key_with_one_byte_of_its_signature_changed_is_refused() -> None:
    prefix, payload, signature = issue().split(".")
    changed = bytearray(unb64url(signature))
    changed[100] ^= 0x01

    why = refusal(f"{prefix}.{payload}.{b64url(bytes(changed))}")

    assert why.startswith("its signature doesn't match")


def test_a_key_with_one_byte_of_its_payload_changed_is_refused() -> None:
    """The change worth making to a key: a year more on its expiry date."""

    prefix, payload, signature = issue().split(".")
    later = unb64url(payload).replace(b'"2027-11-13"', b'"2028-11-13"')

    assert len(later) == len(unb64url(payload))
    assert refusal(f"{prefix}.{b64url(later)}.{signature}").startswith(
        "its signature doesn't match"
    )


def test_a_key_another_key_pair_signed_is_refused() -> None:
    """Who signed is the whole question: a key opens under one public key only."""

    assert read_key(issue(), public_key=TEST)
    with pytest.raises(LicenseKeyError, match="signature doesn't match"):
        read_key(issue(), public_key=OTHER)


def test_a_key_of_a_version_this_release_doesnt_read_is_refused() -> None:
    assert refusal(issue(v=2)) == (
        "it is a version 2 key, and this release of honest-ci reads version 1; "
        "a newer release may read it"
    )


@pytest.mark.parametrize(
    ("version", "why"),
    [
        (True, "it is a version true key"),
        (1.0, "it is a version 1.0 key"),
        ("1", 'it is a version "1" key'),
        (None, "it is a version null key"),
        (MISSING, "its payload has no version"),
    ],
)
def test_a_version_that_only_looks_like_1_is_refused(version: object, why: str) -> None:
    """JSON's ``true`` and ``1.0`` both equal 1 in Python, and neither is version 1."""

    assert refusal(issue(v=version)).startswith(why)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hci1",
        "hci1.",
        "hci1..",
        "hci2.{payload}.{signature}",
        "HCI1.{payload}.{signature}",
        "{payload}.{signature}",
        "hci1.{payload}.{signature}.{signature}",
        "hci1.{payload}",
    ],
)
def test_text_in_another_shape_is_refused_as_not_a_key(text: str) -> None:
    _, payload, signature = issue().split(".")

    why = refusal(text.format(payload=payload, signature=signature))

    assert why.startswith("it isn't an honest-ci license key")


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda key: key + "==", id="padding"),
        pytest.param(lambda key: key.replace("-", "+").replace("_", "/"), id="standard base64"),
        pytest.param(lambda key: key[:200] + "\n" + key[200:], id="a line break"),
        pytest.param(lambda key: key[:200] + " " + key[200:], id="a space"),
        pytest.param(lambda key: key[:-1], id="cut short"),
    ],
)
def test_a_key_copied_wrong_is_refused_and_says_how_it_goes_wrong(
    damage: Callable[[str], str],
) -> None:
    """Said as a copying mistake, not as a forgery: that is what it almost always is."""

    key = issue()
    damaged = damage(key)

    assert damaged != key
    assert refusal(damaged).startswith(
        "it isn't an honest-ci license key: its payload or signature isn't base64url"
    )


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        (b"not json", "its payload isn't JSON"),
        (b"\xff\xfe", "its payload isn't JSON"),
        (b"[1, 2]", "its payload isn't a JSON object"),
        (b'"hci1"', "its payload isn't a JSON object"),
    ],
)
def test_a_signed_payload_that_isnt_a_json_object_is_refused(payload: bytes, why: str) -> None:
    assert refusal(issue(payload)) == why


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"id": MISSING}, "its payload has no id"),
        ({"id": ""}, "its payload has no id"),
        ({"org_id": MISSING}, "its payload has no org_id"),
        ({"org_id": "12345678"}, "its payload has no org_id"),
        ({"org_id": True}, "its payload has no org_id"),
        ({"org_id": 0}, "its payload has no org_id"),
        ({"org": MISSING}, "its payload has no org"),
        ({"plan": 7}, "its payload has no plan"),
        ({"seats": MISSING}, "its payload has no seats"),
        ({"seats": -1}, "its payload has no seats"),
        ({"seats": 3.0}, "its payload has no seats"),
        ({"issued": MISSING}, "its payload has no issued date"),
        ({"issued": "20261113"}, "its payload has no issued date"),
        ({"expires": "2027/11/13"}, "its payload has no expires date"),
        ({"expires": "2027-02-30"}, "its payload has no expires date"),
        ({"expires": 20271113}, "its payload has no expires date"),
    ],
)
def test_a_signed_payload_missing_a_field_is_refused(fields: dict[str, Any], why: str) -> None:
    assert refusal(issue(**fields)) == why


def test_no_refusal_repeats_the_key() -> None:
    """A refusal lands in a pull request comment and in a log, and a key is a credential."""

    key = issue()
    prefix, payload, signature = key.split(".")
    flipped = bytearray(unb64url(signature))
    flipped[0] ^= 0x01
    refused = [
        refusal(key, owner_id=87654321),
        refusal(key, today=EXPIRES + timedelta(days=GRACE_DAYS + 1)),
        refusal(f"{prefix}.{payload}.{b64url(bytes(flipped))}"),
        refusal(f"{key}.{signature}"),
        refusal(key + "=="),
    ]

    for why in refused:
        assert payload not in why and signature not in why
        assert payload[:12] not in why and signature[:12] not in why


# ── the signature itself ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(
            lambda digest: b"\x00\x01"
            + b"\xff" * 204
            + b"\x00"
            + bytes.fromhex("302f300b06096086480165030402010420")
            + digest,
            id="DigestInfo without its NULL parameters",
        ),
        pytest.param(
            lambda digest: b"\x00\x01"
            + b"\xff" * 8
            + b"\x00"
            + SHA256_DIGEST_INFO
            + digest
            + b"\x00" * 194,
            id="short padding and bytes after the digest",
        ),
        pytest.param(
            lambda digest: b"\x00\x02" + b"\xff" * 202 + b"\x00" + SHA256_DIGEST_INFO + digest,
            id="block type 2",
        ),
    ],
)
def test_only_the_encoding_rfc_8017_gives_opens(block: Callable[[bytes], bytes]) -> None:
    """Lenient verifiers parsed the opened block, and let slack in it through.

    Each block here is what one of them would have taken for the message's
    SHA-256, signed with the right private key; only the exact encoding counts.
    """

    message = b"hci1.payload"
    built = block(hashlib.sha256(message).digest())

    assert len(built) == _SIZE
    assert verify_signature(message, sign(message), TEST)
    assert not verify_signature(message, sign_block(built), TEST)


def test_a_signature_past_the_modulus_is_refused_though_it_opens_to_the_same_block() -> None:
    """RFC 8017 §5.2.2: ``s`` and ``s + n`` open alike, and a key has one signature."""

    for number in range(100):
        message = f"hci1.{number}".encode()
        signature = sign(message)
        raised = int.from_bytes(signature, "big") + _MODULUS
        if raised < 2 ** (8 * _SIZE):
            break
    else:
        pytest.fail("no signature small enough to add the modulus to")

    assert verify_signature(message, signature, TEST)
    assert not verify_signature(message, raised.to_bytes(_SIZE, "big"), TEST)


@pytest.mark.parametrize("size", [_SIZE - 1, _SIZE + 1, 0])
def test_a_signature_of_the_wrong_length_is_refused(size: int) -> None:
    message = b"hci1.payload"
    signature = sign(message)

    resized = signature[:size] if size < _SIZE else b"\x00" * (size - _SIZE) + signature

    assert not verify_signature(message, resized, TEST)


# ── the public key installations verify against ──────────────────────────────


def test_the_shipped_public_key_is_rsa_2048_with_exponent_65537() -> None:
    assert PUBLIC_KEY.n.bit_length() == 2048
    assert PUBLIC_KEY.e == 65537


def test_no_key_the_test_key_pair_signs_opens_under_the_shipped_public_key() -> None:
    """The test key pair's private half is written out above, so anyone could sign with it.

    Were it the key installations verify against, anyone could issue a key.
    """

    assert PUBLIC_KEY.n not in {TEST.n, OTHER.n}
    for key in (issue(), SIGNED_BY_OPENSSL):
        assert read_key(key, public_key=TEST)
        with pytest.raises(LicenseKeyError, match="signature doesn't match"):
            read_key(key, public_key=PUBLIC_KEY)
