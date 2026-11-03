import os
import sys

import requests

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 30 

passed = 0
failed = 0


def check(name: str, condition: bool, response: requests.Response) -> None:
    """Record one test result and print it."""
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] {name}")
    print(f"        status = {response.status_code}")
    print(f"        body   = {response.text[:200]}")



def test_health() -> None:
    r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    ok = r.status_code == 200 and r.json().get("status") == "ok"
    check("GET /health  reports ok", ok, r)


def test_predict_positive() -> None:
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"text": "I love this course!"},
        timeout=TIMEOUT,
    )
    body = r.json() if r.status_code == 200 else {}
    ok = (
        r.status_code == 200
        and body.get("label") == "POSITIVE"
        and 0.0 <= body.get("score", -1) <= 1.0
    )
    check("POST /predict  positive text -> POSITIVE", ok, r)


def test_predict_negative() -> None:
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"text": "This was a terrible experience."},
        timeout=TIMEOUT,
    )
    body = r.json() if r.status_code == 200 else {}
    ok = (
        r.status_code == 200
        and body.get("label") == "NEGATIVE"
        and 0.0 <= body.get("score", -1) <= 1.0
    )
    check("POST /predict  negative text -> NEGATIVE", ok, r)


def test_predict_long_text() -> None:
    # 1500 characters: valid for the API, exercises the truncation path.
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"text": "great " * 250},
        timeout=TIMEOUT,
    )
    check("POST /predict  long text is handled", r.status_code == 200, r)


def test_empty_text() -> None:
    r = requests.post(f"{BASE_URL}/predict", json={"text": ""}, timeout=TIMEOUT)
    check("POST /predict  empty text -> 422", r.status_code == 422, r)


def test_blank_text() -> None:
    r = requests.post(f"{BASE_URL}/predict", json={"text": "   "}, timeout=TIMEOUT)
    check("POST /predict  whitespace-only text -> 422", r.status_code == 422, r)


def test_missing_field() -> None:
    r = requests.post(f"{BASE_URL}/predict", json={}, timeout=TIMEOUT)
    check("POST /predict  missing 'text' -> 422", r.status_code == 422, r)


def test_wrong_type() -> None:
    r = requests.post(f"{BASE_URL}/predict", json={"text": 123}, timeout=TIMEOUT)
    check("POST /predict  text is a number -> 422", r.status_code == 422, r)


def test_text_too_long() -> None:
    r = requests.post(
        f"{BASE_URL}/predict", json={"text": "a" * 2001}, timeout=TIMEOUT
    )
    check("POST /predict  text over 2000 chars -> 422", r.status_code == 422, r)


def test_malformed_json() -> None:
    r = requests.post(
        f"{BASE_URL}/predict",
        data="{this is not valid json",
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT,
    )
    check("POST /predict  malformed JSON -> 422", r.status_code == 422, r)


def main() -> int:
    print(f"Testing API at {BASE_URL}\n")
    tests = [
        test_health,
        test_predict_positive,
        test_predict_negative,
        test_predict_long_text,
        test_empty_text,
        test_blank_text,
        test_missing_field,
        test_wrong_type,
        test_text_too_long,
        test_malformed_json,
    ]
    try:
        for test in tests:
            test()
    except requests.exceptions.ConnectionError:
        print(
            f"\nCould not connect to {BASE_URL}.\n"
            "Is the server running?  Start it with:  uvicorn main:app"
        )
        return 2
    except requests.exceptions.Timeout:
        print("\nThe request timed out. The model may still be loading.")
        return 2

    print(f"\nResult: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())