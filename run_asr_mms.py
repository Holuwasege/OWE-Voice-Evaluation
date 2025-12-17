model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

audio_paths_in_merge = data_merge["audio_path"].tolist()
references = data_merge["norm_text"].tolist()

predictions = []

# -------------------------------
# RUN ASR ON ALL AUDIO
# -------------------------------
for audio_file in audio_paths_in_merge:
    try:
        # Load audio
        speech, sr = librosa.load(audio_file, sr=16000)

        # Preprocess
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt").to(model.device)

        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        predictions.append(transcription)

    except Exception as e:
        print(f"Error on file: {audio_file}, error: {e}")
        predictions.append("ERROR")
