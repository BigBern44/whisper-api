from faster_whisper import WhisperModel
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import time


class TritonPythonModel:
    def initialize(self, args):
        self.model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=2,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # ── Lecture des inputs ────────────────────────────────────────
                audio = pb_utils.get_input_tensor_by_name(request, "audio_signal").as_numpy()
                language_raw = pb_utils.get_input_tensor_by_name(request, "language").as_numpy()[0]
                task_raw = pb_utils.get_input_tensor_by_name(request, "task").as_numpy()[0]

                language = language_raw.decode("utf-8") if isinstance(language_raw, bytes) else str(language_raw)
                task = task_raw.decode("utf-8") if isinstance(task_raw, bytes) else str(task_raw)

                if language == "auto":
                    language = None

                # ── Normalisation audio ───────────────────────────────────────
                # max_batch_size: 0 → pas de dim batch, shape = [num_samples]
                audio = audio.flatten().astype(np.float32)

                # ── Inférence ─────────────────────────────────────────────────
                start_time = time.time()
                segments, info = self.model.transcribe(
                    audio,
                    language=language,
                    task=task,
                    without_timestamps=False,
                    vad_filter=True,
                    beam_size=5,
                )
                inference_time = time.time() - start_time

                # ── Construction de la réponse ────────────────────────────────
                result = {
                    "text": " ".join(seg.text.strip() for seg in segments),
                    "language": info.language,
                    "language_probability": round(info.language_probability, 4),
                    "duration": round(info.duration, 3),
                    "inference_time": round(inference_time, 3),
                }

                out_tensor = pb_utils.Tensor(
                    "transcription",
                    np.array([json.dumps(result).encode("utf-8")], dtype=np.object_),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            except Exception as e:
                err = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(output_tensors=[], err=err))

        return responses

    def finalize(self):
        self.model = None
