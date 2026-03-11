from faster_whisper import WhisperModel
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import time


MAX_SAMPLES = 480_000  # 30s × 16 000 Hz


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
                # Avec max_batch_size > 0, Triton ajoute une dim batch :
                # audio_signal : shape [batch, 480000]  → on prend [0] pour traiter 1 par 1
                audio_padded = pb_utils.get_input_tensor_by_name(request, "audio_signal").as_numpy()
                audio_len    = pb_utils.get_input_tensor_by_name(request, "audio_len").as_numpy()
                language_raw = pb_utils.get_input_tensor_by_name(request, "language").as_numpy()
                task_raw     = pb_utils.get_input_tensor_by_name(request, "task").as_numpy()

                batch_size = audio_padded.shape[0]
                results = []

                for i in range(batch_size):
                    # Découpe le signal réel (supprime le padding)
                    real_len = int(audio_len[i][0])
                    audio    = audio_padded[i, :real_len].astype(np.float32)

                    language = language_raw[i][0]
                    language = language.decode("utf-8") if isinstance(language, bytes) else str(language)
                    if language == "auto":
                        language = None

                    task = task_raw[i][0]
                    task = task.decode("utf-8") if isinstance(task, bytes) else str(task)

                    # ── Inférence ─────────────────────────────────────────────
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

                    results.append({
                        "text": " ".join(seg.text.strip() for seg in segments),
                        "language": info.language,
                        "language_probability": round(info.language_probability, 4),
                        "duration": round(info.duration, 3),
                        "inference_time": round(inference_time, 3),
                    })

                # ── Réponse batch ─────────────────────────────────────────────
                out = np.array(
                    [[json.dumps(r).encode("utf-8")] for r in results],
                    dtype=np.object_,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("transcription", out)]
                    )
                )

            except Exception as e:
                err = pb_utils.TritonError(str(e))
                responses.append(pb_utils.InferenceResponse(output_tensors=[], err=err))

        return responses

    def finalize(self):
        self.model = None
