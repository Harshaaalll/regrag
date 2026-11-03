from regrag.guardrails import is_advice_request, is_off_topic, mask_pii


def test_masks_each_pii_type():
    t, found = mask_pii("PAN ABCDE1234F, Aadhaar 1234 5678 9012, call 9876543210, acct 123456789012345")
    assert "ABCDE1234F" not in t and "9876543210" not in t and "1234 5678 9012" not in t and "123456789012345" not in t
    assert set(found) == {"PAN", "AADHAAR", "PHONE", "ACCOUNT"}


def test_plain_text_untouched():
    t, found = mask_pii("What is the limit of Rs 50,000 for a small account in 2026?")
    assert t.startswith("What is the limit") and found == []


def test_off_topic():
    assert is_off_topic("Who won the cricket world cup?")
    assert is_off_topic("Write a Python script to sort a list")
    assert not is_off_topic("What are the limits of a Small Account?")
    assert not is_off_topic("How long must records be preserved?")


def test_advice_detection():
    assert is_advice_request("Should my bank freeze this customer's account?")
    assert is_advice_request("Can I avoid KYC by splitting deposits?")
    assert not is_advice_request("What does paragraph 28 say about small accounts?")
