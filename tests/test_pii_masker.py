"""Tests for PII masker."""

from __future__ import annotations

from src.pii.masker import PIIMasker


class TestPIIMasker:
    """Tests for PII detection and redaction."""

    def setup_method(self):
        self.masker = PIIMasker(enabled=True)

    def test_masks_aadhaar(self):
        """12-digit Aadhaar numbers should be replaced."""
        text = "My Aadhaar is 1234 5678 9012"
        result = self.masker.mask(text)
        assert "[AADHAAR]" in result
        assert "1234" not in result

    def test_masks_pan(self):
        """PAN card format ABCDE1234F should be replaced."""
        text = "PAN number: ABCPK1234F"
        result = self.masker.mask(text)
        assert "[PAN]" in result
        assert "ABCPK1234F" not in result

    def test_masks_phone(self):
        """Indian phone numbers should be replaced."""
        text = "Call me at +91 9876543210"
        result = self.masker.mask(text)
        assert "[PHONE]" in result
        assert "9876543210" not in result

    def test_masks_email(self):
        """Email addresses should be replaced."""
        text = "Send to sharma.raj@example.com please"
        result = self.masker.mask(text)
        assert "[EMAIL]" in result
        assert "sharma.raj@example.com" not in result

    def test_disabled_masker_passthrough(self):
        """When disabled, text should pass through unchanged."""
        masker = PIIMasker(enabled=False)
        text = "PAN: ABCPK1234F, Phone: 9876543210"
        assert masker.mask(text) == text

    def test_empty_string(self):
        """Empty strings should return empty."""
        assert self.masker.mask("") == ""

    def test_no_pii(self):
        """Text without PII should pass through unchanged."""
        text = "Hello, how can I help you today?"
        assert self.masker.mask(text) == text

    def test_mask_turns(self):
        """mask_turns should mask transcript field in turn dicts."""
        turns = [
            {"turn_id": "1", "transcript": "My phone is 9876543210"},
            {"turn_id": "2", "transcript": "Hello there"},
        ]
        masked = self.masker.mask_turns(turns)
        assert "[PHONE]" in masked[0]["transcript"]
        assert masked[1]["transcript"] == "Hello there"
        # Originals should not be mutated
        assert "9876543210" in turns[0]["transcript"]
