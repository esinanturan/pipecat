from pipecat.services.google.tts import GeminiTTSService


def test_gemini_tts_available_voices_match_documented_names():
    voices = GeminiTTSService.AVAILABLE_VOICES

    assert "Callirrhoe" in voices
    assert "Sulafat" in voices
    assert "Callirhoe" not in voices
    assert "Sulafar" not in voices
