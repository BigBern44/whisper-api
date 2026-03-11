import io
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from faster_whisper import WhisperModel


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        
        params = model_config.get("parameters", {})
        whisper_model = params.get("WHISPER_MODEL", {}).get("string_value", "tiny")
        whisper_device = params.get("WHISPER_DEVICE", {}).get("string_value", "cpu")
        whisper_compute_type = params.get("WHISPER_COMPUTE_TYPE", {}).get("string_value", "int8")
        
        print(f"Loading Whisper model: {whisper_model} on {whisper_device}")
        self.model = WhisperModel(
            whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type
        )
        print("Whisper model loaded successfully")

    def execute(self, requests):
        responses = []
        
        for request in requests:
            audio_bytes_tensor = pb_utils.get_input_tensor_by_name(request, "audio_bytes")
            audio_bytes = audio_bytes_tensor.as_numpy()[0]  # bytes

            language = None
            language_tensor = pb_utils.get_input_tensor_by_name(request, "language")
            if language_tensor is not None:
                lang_value = language_tensor.as_numpy()[0]
                if isinstance(lang_value, bytes):
                    lang_value = lang_value.decode("utf-8")
                if lang_value and lang_value != "auto":
                    language = lang_value

            try:
                audio_buffer = io.BytesIO(audio_bytes)

                segments, info = self.model.transcribe(
                    audio_buffer,
                    language=language,
                    beam_size=5,
                    vad_filter=True
                )

                result_segments = []
                full_text = []
                for segment in segments:
                    result_segments.append({
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                        "text": segment.text.strip()
                    })
                    full_text.append(segment.text.strip())

                result = {
                    "status": "completed",
                    "text": " ".join(full_text),
                    "segments": result_segments,
                    "language": info.language,
                    "language_probability": round(info.language_probability, 3),
                    "duration": round(info.duration, 2)
                }

            except Exception as e:
                result = {
                    "status": "failed",
                    "error": str(e)
                }

            output_tensor = pb_utils.Tensor(
                "transcription",
                np.array([json.dumps(result)], dtype=object)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        
        return responses

    def finalize(self):
        print("Whisper model unloaded")