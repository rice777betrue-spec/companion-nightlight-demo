from __future__ import annotations

import math
import threading
import wave
from pathlib import Path

import numpy as np

from companion_demo.core.contracts import (
    SpeakerVerification,
    VoiceprintEnrollment,
)


class LocalVoiceprintAdapter:
    """无需额外模型的电脑端声纹基线。

    它从语音中提取倒谱、频谱和基频统计量并进行余弦匹配，适合验证产品
    链路与权限策略。量产设备应通过同一端口替换为 ECAPA/ResNet 声纹模型。
    """

    _TARGET_SAMPLE_RATE = 16_000
    _PROFILE_VERSION = 1

    def __init__(
        self,
        profile_path: Path,
        *,
        threshold: float = 0.82,
        required_samples: int = 3,
        maximum_samples: int = 8,
        minimum_seconds: float = 1.2,
    ) -> None:
        self.profile_path = Path(profile_path)
        self.threshold = max(-1.0, min(1.0, float(threshold)))
        self.required_samples = max(1, int(required_samples))
        self.maximum_samples = max(
            self.required_samples,
            int(maximum_samples),
        )
        self.minimum_seconds = max(0.5, float(minimum_seconds))
        self._lock = threading.RLock()
        self._embeddings = np.empty((0, 0), dtype=np.float32)
        self._owner_name = ""
        self._status_text = "尚未录入主人声纹"
        self._mel_filters = self._build_mel_filters()
        self._dct = self._build_dct(20, 32)
        self._load_profile()

    @property
    def status_text(self) -> str:
        with self._lock:
            return self._status_text

    @property
    def sample_count(self) -> int:
        with self._lock:
            return int(self._embeddings.shape[0])

    @property
    def ready(self) -> bool:
        return self.sample_count >= self.required_samples

    @staticmethod
    def _hz_to_mel(value: float) -> float:
        return 2595.0 * math.log10(1.0 + value / 700.0)

    @staticmethod
    def _mel_to_hz(value: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10.0, value / 2595.0) - 1.0)

    def _build_mel_filters(self) -> np.ndarray:
        sample_rate = self._TARGET_SAMPLE_RATE
        fft_size = 512
        mel_points = np.linspace(
            self._hz_to_mel(80.0),
            self._hz_to_mel(sample_rate / 2),
            34,
        )
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
        bins = np.clip(bins, 0, fft_size // 2)
        filters = np.zeros((32, fft_size // 2 + 1), dtype=np.float32)
        for index in range(32):
            left, center, right = bins[index : index + 3]
            center = max(center, left + 1)
            right = max(right, center + 1)
            right = min(right, fft_size // 2)
            for position in range(left, min(center, filters.shape[1])):
                filters[index, position] = (position - left) / (center - left)
            for position in range(center, min(right + 1, filters.shape[1])):
                filters[index, position] = (right - position) / (right - center)
        normalizer = filters.sum(axis=1, keepdims=True)
        return filters / np.maximum(normalizer, 1e-8)

    @staticmethod
    def _build_dct(coefficients: int, bands: int) -> np.ndarray:
        band_positions = np.arange(bands, dtype=np.float32) + 0.5
        coefficient_positions = np.arange(coefficients, dtype=np.float32)[:, None]
        matrix = np.cos(np.pi * coefficient_positions * band_positions / bands)
        matrix[0] *= math.sqrt(1.0 / bands)
        matrix[1:] *= math.sqrt(2.0 / bands)
        return matrix.astype(np.float32)

    @staticmethod
    def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
        if sample_width == 1:
            return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        if sample_width == 2:
            return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if sample_width == 3:
            packed = np.frombuffer(raw, dtype=np.uint8)
            if packed.size % 3:
                raise ValueError("24 位 WAV 数据长度无效")
            packed = packed.reshape(-1, 3).astype(np.int32)
            values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
            values = np.where(values & 0x800000, values - 0x1000000, values)
            return values.astype(np.float32) / 8_388_608.0
        if sample_width == 4:
            return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
        raise ValueError(f"暂不支持 {sample_width * 8} 位 WAV")

    def _load_audio(self, audio_path: str) -> np.ndarray:
        path = Path(audio_path)
        if not path.is_file():
            raise ValueError("找不到用于声纹处理的音频")
        try:
            with wave.open(str(path), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sample_rate = source.getframerate()
                frames = source.readframes(source.getnframes())
        except (wave.Error, EOFError) as exc:
            raise ValueError("声纹录入目前需要 WAV 音频") from exc

        samples = self._decode_pcm(frames, sample_width)
        if channels > 1:
            usable = samples[: samples.size - (samples.size % channels)]
            samples = usable.reshape(-1, channels).mean(axis=1)
        if sample_rate != self._TARGET_SAMPLE_RATE and samples.size:
            target_length = max(
                1,
                int(round(samples.size * self._TARGET_SAMPLE_RATE / sample_rate)),
            )
            source_points = np.linspace(0.0, 1.0, samples.size, endpoint=False)
            target_points = np.linspace(0.0, 1.0, target_length, endpoint=False)
            samples = np.interp(target_points, source_points, samples).astype(np.float32)
        maximum = int(self._TARGET_SAMPLE_RATE * 15)
        return np.asarray(samples[:maximum], dtype=np.float32)

    @staticmethod
    def _pitch_features(frames: np.ndarray, energies: np.ndarray) -> np.ndarray:
        strongest = np.argsort(energies)[-min(80, frames.shape[0]) :]
        pitches: list[float] = []
        confidences: list[float] = []
        minimum_lag = 50
        maximum_lag = min(267, frames.shape[1] - 2)
        for index in strongest:
            frame = frames[index] - float(np.mean(frames[index]))
            energy = float(np.dot(frame, frame))
            if energy <= 1e-8:
                continue
            correlation = np.correlate(frame, frame, mode="full")[frame.size - 1 :]
            search = correlation[minimum_lag : maximum_lag + 1]
            if search.size == 0:
                continue
            offset = int(np.argmax(search))
            lag = minimum_lag + offset
            confidence = float(search[offset] / max(correlation[0], 1e-8))
            if confidence < 0.18:
                continue
            pitches.append(16_000.0 / lag)
            confidences.append(confidence)
        if not pitches:
            return np.zeros(6, dtype=np.float32)
        values = np.log(np.asarray(pitches, dtype=np.float32))
        confidence_values = np.asarray(confidences, dtype=np.float32)
        return np.asarray(
            [
                values.mean(),
                values.std(),
                np.percentile(values, 25),
                np.percentile(values, 75),
                confidence_values.mean(),
                confidence_values.std(),
            ],
            dtype=np.float32,
        )

    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        samples = self._load_audio(audio_path)
        minimum_samples = int(self._TARGET_SAMPLE_RATE * self.minimum_seconds)
        if samples.size < minimum_samples:
            raise ValueError(
                f"声纹音频太短，请连续说话至少 {self.minimum_seconds:.1f} 秒"
            )
        samples = samples - float(np.mean(samples))
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms < 0.003:
            raise ValueError("录音声音太小，请靠近麦克风重新录入")
        peak = float(np.max(np.abs(samples)))
        samples = samples / max(peak, 1e-6)
        samples[1:] = samples[1:] - 0.97 * samples[:-1]

        frame_length = 400
        hop_length = 160
        frame_count = 1 + (samples.size - frame_length) // hop_length
        if frame_count < 4:
            raise ValueError("有效语音不足，无法提取声纹")
        frames = np.lib.stride_tricks.sliding_window_view(samples, frame_length)[
            ::hop_length
        ][:frame_count].copy()
        energies = np.mean(frames * frames, axis=1)
        energy_db = 10.0 * np.log10(np.maximum(energies, 1e-10))
        voice_threshold = max(
            float(np.max(energy_db) - 38.0),
            float(np.percentile(energy_db, 25)),
        )
        voiced = frames[energy_db >= voice_threshold]
        voiced_energies = energies[energy_db >= voice_threshold]
        if voiced.shape[0] < 20:
            raise ValueError("有效语音太少，请说一段完整句子")

        pitch = self._pitch_features(voiced, voiced_energies)
        windowed = voiced * np.hamming(frame_length).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(windowed, n=512, axis=1)) ** 2
        mel_energy = spectrum @ self._mel_filters.T
        log_mel = np.log(np.maximum(mel_energy, 1e-10)).astype(np.float32)
        cepstrum = log_mel @ self._dct.T
        cepstrum = cepstrum[:, 1:]
        deltas = np.diff(cepstrum, axis=0)

        features = np.concatenate(
            [
                cepstrum.mean(axis=0) / 6.0,
                cepstrum.std(axis=0) / 4.0,
                np.percentile(cepstrum, 25, axis=0) / 6.0,
                np.percentile(cepstrum, 75, axis=0) / 6.0,
                deltas.mean(axis=0) / 3.0,
                deltas.std(axis=0) / 3.0,
                pitch / np.asarray([6.0, 1.0, 6.0, 6.0, 1.0, 1.0], dtype=np.float32),
            ]
        ).astype(np.float32)
        norm = float(np.linalg.norm(features))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("声纹特征无效，请重新录音")
        return features / norm

    def _load_profile(self) -> None:
        with self._lock:
            if not self.profile_path.is_file():
                self._status_text = (
                    f"尚未录入主人声纹（0/{self.required_samples}）"
                )
                return
            try:
                with np.load(self.profile_path, allow_pickle=False) as profile:
                    version = int(profile["version"][0])
                    embeddings = np.asarray(profile["embeddings"], dtype=np.float32)
                    owner_name = str(profile["owner_name"][0])
                if version != self._PROFILE_VERSION or embeddings.ndim != 2:
                    raise ValueError("声纹档案版本不兼容")
                if embeddings.shape[0] and embeddings.shape[1] == 0:
                    raise ValueError("声纹档案为空")
                self._embeddings = embeddings
                self._owner_name = owner_name
                self._status_text = self._profile_status()
            except Exception as exc:
                self._embeddings = np.empty((0, 0), dtype=np.float32)
                self._owner_name = ""
                self._status_text = f"声纹档案读取失败：{exc}"

    def _profile_status(self) -> str:
        count = int(self._embeddings.shape[0])
        owner = f"（{self._owner_name}）" if self._owner_name else ""
        if count >= self.required_samples:
            return f"主人声纹{owner}已就绪（{count} 段）"
        return f"正在录入主人声纹{owner}（{count}/{self.required_samples}）"

    def _save_profile(self) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.profile_path.with_suffix(self.profile_path.suffix + ".tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                version=np.asarray([self._PROFILE_VERSION], dtype=np.int16),
                embeddings=self._embeddings.astype(np.float32),
                owner_name=np.asarray([self._owner_name]),
            )
        temporary.replace(self.profile_path)

    def enroll(
        self,
        audio_path: str,
        owner_name: str = "",
    ) -> VoiceprintEnrollment:
        if not audio_path:
            raise ValueError("请先录制一段主人语音")
        embedding = self._extract_embedding(audio_path)
        with self._lock:
            if self._embeddings.size:
                similarities = self._embeddings @ embedding
                similarity = float(np.max(similarities))
                if similarity > 0.99995:
                    raise ValueError(
                        "这段录音已经录入，请重新说一段不同句子"
                    )
                if similarity < 0.45:
                    raise ValueError(
                        "这段声音与前面的录入差异过大，请确认由同一人录制"
                    )
                self._embeddings = np.vstack([self._embeddings, embedding])
            else:
                self._embeddings = embedding.reshape(1, -1)
            if self._embeddings.shape[0] > self.maximum_samples:
                self._embeddings = self._embeddings[-self.maximum_samples :]
            if owner_name.strip():
                self._owner_name = owner_name.strip()
            self._save_profile()
            self._status_text = self._profile_status()
            count = int(self._embeddings.shape[0])
            return VoiceprintEnrollment(
                sample_count=count,
                required_samples=self.required_samples,
                ready=count >= self.required_samples,
                status=self._status_text,
            )

    @staticmethod
    def _normalized_mean(embeddings: np.ndarray) -> np.ndarray:
        centroid = embeddings.mean(axis=0)
        return centroid / max(float(np.linalg.norm(centroid)), 1e-8)

    def verify(self, audio_path: str) -> SpeakerVerification:
        with self._lock:
            count = int(self._embeddings.shape[0])
            ready = count >= self.required_samples
            embeddings = self._embeddings.copy()
        if not ready:
            status = self._profile_status()
            with self._lock:
                self._status_text = status
            return SpeakerVerification(
                identity="not_enrolled",
                enrolled=False,
                is_owner=None,
                score=None,
                threshold=self.threshold,
                sample_count=count,
                status=status,
            )
        try:
            embedding = self._extract_embedding(audio_path)
            centroid = self._normalized_mean(embeddings)
            centroid_score = float(np.dot(centroid, embedding))
            sample_score = float(np.median(embeddings @ embedding))
            score = 0.7 * centroid_score + 0.3 * sample_score
            is_owner = score >= self.threshold
            identity = "owner" if is_owner else "guest"
            label = "主人 ✓" if is_owner else "访客"
            status = (
                f"声纹：{label}｜相似度 {score * 100:.0f}%"
                f"｜阈值 {self.threshold * 100:.0f}%"
            )
        except Exception as exc:
            score = None
            is_owner = False
            identity = "unverified"
            status = f"声纹：无法确认身份（{exc}）"
        with self._lock:
            self._status_text = status
        return SpeakerVerification(
            identity=identity,
            enrolled=True,
            is_owner=is_owner,
            score=score,
            threshold=self.threshold,
            sample_count=count,
            status=status,
        )

    def clear(self) -> str:
        with self._lock:
            self.profile_path.unlink(missing_ok=True)
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            self._owner_name = ""
            self._status_text = (
                f"主人声纹已删除（0/{self.required_samples}），可重新录入"
            )
            return self._status_text
