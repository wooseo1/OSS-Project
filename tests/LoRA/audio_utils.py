"""오디오 로드 유틸: 어떤 WAV든 16kHz mono float32로 통일"""
import numpy as np

TARGET_SR = 16000


def load_audio(path, speed=1.0):
    """WAV를 16kHz mono float32 배열로 로드. speed != 1.0 이면 속도 변형(피치 포함)"""
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != TARGET_SR or speed != 1.0:
        import librosa

        # speed > 1: 빨라짐. orig_sr 를 키워서 리샘플링하면 길이가 1/speed 로 줄어듦
        y = librosa.resample(y, orig_sr=int(sr * speed), target_sr=TARGET_SR)
    return np.ascontiguousarray(y, dtype=np.float32)


def audio_duration(path):
    """파일 헤더만 읽어 길이(초) 반환"""
    import soundfile as sf

    return float(sf.info(str(path)).duration)
