from flask import Flask, request, jsonify
import numpy as np
from scipy.signal import find_peaks, detrend, butter, filtfilt, savgol_filter
from scipy.stats import zscore

app = Flask(__name__)

def bandpass_filter(signal, lowcut=0.7, highcut=4.0, fs=30.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def refine_signal(raw_signal, fps):
    signal = detrend(raw_signal)
    signal = bandpass_filter(signal, fs=fps)
    signal = savgol_filter(signal, window_length=9, polyorder=2)
    signal = zscore(signal)
    return signal.tolist()

def estimate_heart_rate(brightness_values, fps):
    window_size = 5
    refined = refine_signal(brightness_values, fps)
    smoothed = np.convolve(refined, np.ones(window_size)/window_size, mode='valid')
    smoothed -= np.mean(smoothed)

    peaks, _ = find_peaks(smoothed, distance=fps * 0.45)
    if len(peaks) < 2:
        return None

    intervals = np.diff(peaks) / fps
    avg_interval = np.mean(intervals)
    bpm = 60 / avg_interval
    return round(bpm, 2)

@app.route("/heartrate", methods=["POST"])
def heartrate():
    data = request.get_json()
    brightness = data.get("brightness")
    fps = data.get("fps", 30.0)

    if not brightness:
        return jsonify({"error": "Missing brightness data"}), 400

    hr = estimate_heart_rate(brightness, fps)
    return jsonify({"bpm": hr})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
