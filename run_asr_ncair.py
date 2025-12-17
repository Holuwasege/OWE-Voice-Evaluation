processor = WhisperProcessor.from_pretrained("NCAIR1/Yoruba-ASR")
model = WhisperForConditionalGeneration.from_pretrained("NCAIR1/Yoruba-ASR")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
audio_paths_in_merge = datasets["audio_path"].tolist()
references = datasets["norm_text"].tolist()

results = []

for audio_files in audio_paths_in_merge:
  try:
    # Load audio
    audio, sr = librosa.load(audio_files, sr=16000)

    # Process
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(input_features, max_length=200,)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    results.append(text)

  except Exception as e:
      print(f"Error on file: {audio_files}, error: {e}")
      results.append("ERROR")
