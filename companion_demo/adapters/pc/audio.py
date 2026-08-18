from __future__ import annotations

import queue
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad

from companion_demo.core.contracts import AudioFrame, VadDecision

try:
    import winsound
except ImportError:  # pragma: no cover - 仅非 Windows 开发环境会走到这里
    winsound = None


class SoundDeviceAudioInput:
    """电脑麦克风适配器。

    PortAudio 回调只把 20 ms PCM 帧写入固定队列；消费变慢时丢弃最旧帧，
    因而不会让内存随运行时间增长。
    """

    def __init__(
        self,
        *,
        device: str | int | None = None,
        sample_rate: int = 16_000,
        frame_duration_ms: int = 20,
        queue_frames: int = 100,
    ) -> None:
        self.device = device
        self.sample_rate = int(sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_samples = int(
            self.sample_rate * self.frame_duration_ms / 1000
        )
        self._frames: queue.Queue[AudioFrame] = queue.Queue(
            maxsize=max(10, int(queue_frames))
        )
        self._stream: sd.InputStream | None = None
        self._lock = threading.RLock()
        self._device_label = "系统默认麦克风"
        self._dropped_frames = 0

    @property
    def device_label(self) -> str:
        with self._lock:
            return self._device_label

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames

    def _discard_queued_frames(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _audio_callback(self, indata, _frames, _time_info, status) -> None:
        frame = AudioFrame(
            pcm_s16le=bytes(indata),
            sample_rate=self.sample_rate,
            captured_at=time.monotonic(),
        )
        if status.input_overflow:
            with self._lock:
                self._dropped_frames += 1
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                pass
            with self._lock:
                self._dropped_frames += 1

    def start(self) -> None:
        with self._lock:
            if self._stream is not None and self._stream.active:
                return
            self._discard_queued_frames()
            self._dropped_frames = 0
            stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.frame_samples,
                latency="low",
                callback=self._audio_callback,
            )
            try:
                stream.start()
            except Exception:
                stream.close()
                raise
            self._stream = stream
            device_info = sd.query_devices(stream.device, "input")
            self._device_label = str(device_info.get("name", self._device_label))

    def read_frame(self, timeout: float = 0.1) -> AudioFrame | None:
        try:
            return self._frames.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        self._discard_queued_frames()

    def close(self) -> None:
        self.stop()


class AdaptiveEnergyVad:
    """适合首版 Demo 的自适应能量 VAD，不占用额外模型内存。"""

    def __init__(
        self,
        *,
        minimum_rms: float = 500.0,
        noise_multiplier: float = 3.2,
        sensitivity: float = 1.0,
        calibration_frames: int = 50,
        adaptation_rate: float = 0.025,
        noise_window_frames: int = 100,
    ) -> None:
        self.minimum_rms = float(minimum_rms)
        self.noise_multiplier = float(noise_multiplier)
        self.calibration_frames = max(0, int(calibration_frames))
        self.adaptation_rate = max(0.001, min(1.0, float(adaptation_rate)))
        self._sensitivity = 1.0
        self._noise_floor = max(1.0, self.minimum_rms / self.noise_multiplier)
        self._calibration_remaining = self.calibration_frames
        self._recent_levels: deque[float] = deque(
            maxlen=max(20, int(noise_window_frames))
        )
        self._lock = threading.RLock()
        self.set_sensitivity(sensitivity)

    def reset(self) -> None:
        with self._lock:
            self._noise_floor = max(
                1.0, self.minimum_rms / self.noise_multiplier
            )
            self._calibration_remaining = self.calibration_frames
            self._recent_levels.clear()

    def set_sensitivity(self, value: float) -> None:
        with self._lock:
            self._sensitivity = max(0.5, min(2.0, float(value)))

    @staticmethod
    def _rms(frame: AudioFrame) -> float:
        samples = np.frombuffer(frame.pcm_s16le, dtype="<i2")
        if samples.size == 0:
            return 0.0
        floating = samples.astype(np.float32)
        return float(np.sqrt(np.mean(floating * floating)))

    def analyze(self, frame: AudioFrame) -> VadDecision:
        level = self._rms(frame)
        with self._lock:
            self._recent_levels.append(level)
            if self._calibration_remaining > 0:
                calibration_rate = max(self.adaptation_rate, 0.18)
                self._noise_floor = (
                    (1.0 - calibration_rate) * self._noise_floor
                    + calibration_rate * level
                )
                self._calibration_remaining -= 1
                threshold = max(
                    self.minimum_rms,
                    self._noise_floor * self.noise_multiplier,
                ) / self._sensitivity
                return VadDecision(
                    is_speech=False,
                    level=level,
                    threshold=threshold,
                    calibrating=True,
                )

            ordered_levels = sorted(self._recent_levels)
            baseline_index = int((len(ordered_levels) - 1) * 0.2)
            robust_floor = ordered_levels[baseline_index]
            robust_rate = max(self.adaptation_rate, 0.08)
            self._noise_floor = (
                (1.0 - robust_rate) * self._noise_floor
                + robust_rate * robust_floor
            )

            threshold = max(
                self.minimum_rms,
                self._noise_floor * self.noise_multiplier,
            ) / self._sensitivity
            is_speech = level >= threshold
            if not is_speech:
                self._noise_floor = (
                    (1.0 - self.adaptation_rate) * self._noise_floor
                    + self.adaptation_rate * level
                )
            return VadDecision(
                is_speech=is_speech,
                level=level,
                threshold=threshold,
            )


class WebRtcEnergyVad:
    """WebRTC 语音判定与自适应能量门限的双重过滤。"""

    def __init__(
        self,
        *,
        aggressiveness: int = 2,
        energy_vad: AdaptiveEnergyVad | None = None,
    ) -> None:
        self._webrtc = webrtcvad.Vad(max(0, min(3, int(aggressiveness))))
        self._energy = energy_vad or AdaptiveEnergyVad()

    def reset(self) -> None:
        self._energy.reset()

    def set_sensitivity(self, value: float) -> None:
        self._energy.set_sensitivity(value)

    def analyze(self, frame: AudioFrame) -> VadDecision:
        energy = self._energy.analyze(frame)
        if energy.calibrating or not energy.is_speech:
            return energy
        try:
            is_speech = self._webrtc.is_speech(
                frame.pcm_s16le,
                frame.sample_rate,
            )
        except Exception:
            is_speech = False
        return VadDecision(
            is_speech=is_speech,
            level=energy.level,
            threshold=energy.threshold,
        )


class WindowsWavePlayer:
    """使用 Windows 原生音频播放，可从采集线程立即停止以支持打断。"""

    def __init__(self) -> None:
        self._playing = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

    @property
    def engine_label(self) -> str:
        return "Windows 本地扬声器"

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def play(self, audio_path: str) -> None:
        if winsound is None:
            raise RuntimeError("当前系统不支持 Windows WAV 播放")
        path = Path(audio_path)
        if path.suffix.lower() != ".wav":
            raise ValueError("本地自动播放目前需要 WAV 格式的 TTS 输出")
        with wave.open(str(path), "rb") as source:
            duration = source.getnframes() / max(1, source.getframerate())
        with self._lock:
            self._stop_event.clear()
            self._playing = True
        try:
            winsound.PlaySound(
                str(path),
                winsound.SND_FILENAME
                | winsound.SND_NODEFAULT
                | winsound.SND_ASYNC,
            )
            deadline = time.monotonic() + duration + 0.15
            while time.monotonic() < deadline:
                if self._stop_event.wait(timeout=0.02):
                    break
        finally:
            with self._lock:
                self._playing = False

    def stop(self) -> None:
        self._stop_event.set()
        if winsound is None:
            return
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except RuntimeError:
            pass
        finally:
            with self._lock:
                self._playing = False
