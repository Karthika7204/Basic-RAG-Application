from faster_whisper import WhisperModel

# Load the model — can use "tiny", "base", "small", "medium", "large"
model = WhisperModel("tiny", device="cpu")

# Transcribe the audio
segments, info = model.transcribe(r"D:\NIT Intern\RagBased_application_python\data\250625_LEFTN_Israel_Iran_ceasefire_download.mp3")

# Merge all the segments into a single string
audio_text = " ".join([segment.text for segment in segments])

print("Transcribed Text:", audio_text)