import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import pywt
from sklearn.decomposition import FastICA
from scipy.signal import lfilter
from tqdm import tqdm
import math
import os
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import json


# ========== 数据预处理模块 ==========
def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def random_signal(signal, combin_num):
    random_result = []
    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
        random_result.append(shuffled_dataset)
    return np.array(random_result)


def prepare_data(EEG_all, noise_all, combin_num, train_per):
    EEG_all_random = np.squeeze(random_signal(signal=EEG_all, combin_num=1))
    noise_all_random = np.squeeze(random_signal(signal=noise_all, combin_num=1))

    EEG_all_random = EEG_all_random[0:noise_all_random.shape[0]]

    timepoint = noise_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0])
    validation_num = round((EEG_all_random.shape[0] - train_num) / 2)
    test_num = EEG_all_random.shape[0] - train_num - validation_num

    train_eeg = EEG_all_random[0:train_num, :]
    validation_eeg = EEG_all_random[train_num:train_num + validation_num, :]
    test_eeg = EEG_all_random[train_num + validation_num:, :]

    train_noise = noise_all_random[0:train_num, :]
    validation_noise = noise_all_random[train_num:train_num + validation_num, :]
    test_noise = noise_all_random[train_num + validation_num:, :]

    # 测试数据构建
    EEG_test = []
    noise_EEG_test = []
    for j in range(test_eeg.shape[0]):
        eeg = test_eeg[j]
        noise = test_noise[j]
        coe = get_rms(eeg) / (get_rms(noise) * 1.0)  # 固定 SNR = 1
        noise = noise * coe
        neeg = noise + eeg
        noise_EEG_test.append(neeg)
        EEG_test.append(eeg)

    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)

    # 标准化
    EEG_test_end_standard = EEG_test / np.std(noise_EEG_test, axis=1, keepdims=True)
    noiseEEG_test_end_standard = noise_EEG_test / np.std(noise_EEG_test, axis=1, keepdims=True)

    return noiseEEG_test_end_standard, EEG_test_end_standard


# ========== 传统去噪方法 ==========

def fourier_filtering(signal, cutoff_frequency=30, sampling_rate=128):
    signal_fft = fft(signal)
    n = len(signal)
    T = 1.0 / sampling_rate
    xf = np.fft.fftfreq(n, T)

    filter_mask = np.ones_like(signal_fft)
    filter_mask[np.abs(xf) > cutoff_frequency] = 0
    filtered_signal_fft = signal_fft * filter_mask
    filtered_signal = ifft(filtered_signal_fft).real
    return filtered_signal


def wavelet_denoising(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal))) * np.std(coeffs[-1])
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal


def ica_denoising(signal, n_components=1):
    signal_2d = np.expand_dims(signal, axis=0)
    ica = FastICA(n_components=n_components, whiten='unit-variance')
    ica.fit(signal_2d.T)
    ica_components = ica.transform(signal_2d.T)
    denoised_signal = ica_components[:, 0]
    return denoised_signal


def adaptive_filtering(signal, filter_order=10, alpha=0.01):
    filter_coeffs = np.zeros(filter_order)
    filtered_signal = np.zeros_like(signal)
    for i in range(filter_order, len(signal)):
        error = signal[i] - np.dot(filter_coeffs, signal[i - filter_order:i])
        filtered_signal[i] = signal[i] - error
        filter_coeffs += alpha * error * signal[i - filter_order:i]
    return filtered_signal


# ========== 性能指标计算函数 ==========

def calculate_mse(clean, denoised):
    """均方误差"""
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)
    return np.mean((clean - denoised) ** 2)


def calculate_mae(clean, denoised):
    """平均绝对误差"""
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)
    return np.mean(np.abs(clean - denoised))


def calculate_snr(clean, denoised):
    """信噪比"""
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)
    noise = clean - denoised
    snr = 10 * np.log10((np.var(clean) + 1e-10) / (np.var(noise) + 1e-10))
    return snr


def calculate_snr_improvement(clean, noisy, denoised):
    """SNR 改善"""
    input_snr = calculate_snr(clean, noisy)
    output_snr = calculate_snr(clean, denoised)
    return output_snr - input_snr


def calculate_pearson(clean, denoised):
    """皮尔逊相关系数"""
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)

    correlations = []
    for i in range(clean.shape[0]):
        try:
            corr, _ = pearsonr(clean[i], denoised[i])
            correlations.append(corr)
        except:
            continue
    return np.nanmean(correlations)


def waveform_fidelity(clean, denoised):
    """波形保真度"""
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)

    fidelity = []
    for i in range(clean.shape[0]):
        try:
            wf = np.corrcoef(clean[i], denoised[i])[0, 1]
            fidelity.append(wf)
        except:
            continue
    return np.nanmean(fidelity)


def calculate_ssim_score(clean, denoised):
    """结构相似性（SSIM）"""
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)

    scores = []
    for i in range(clean.shape[0]):
        try:
            score = ssim(clean[i], denoised[i], win_size=64)
            scores.append(score)
        except:
            continue
    return np.nanmean(scores)


# ========== 主程序 ==========

def run_experiment():
    result_dir = "results_traditional"
    os.makedirs(result_dir, exist_ok=True)

    file_location = r"D:\EEGdenoiseNet-master\EEGdenoiseNet-master\data\\"
    EEG_all = np.load(file_location + "EEG_all_epochs.npy")
    noise_all = np.load(file_location + "EOG_all_epochs.npy")

    datanum = 512

    # 准备测试集
    X_test, y_test = prepare_data(EEG_all, noise_all, combin_num=10, train_per=0.8)

    # 扩展为模型输入格式（可选）
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    # 存储结果
    results = {}

    # 定义方法字典
    methods = {
        "Fourier": lambda sig: fourier_filtering(sig),
        "Wavelet": lambda sig: wavelet_denoising(sig),
        "ICA": lambda sig: ica_denoising(sig),
        "AdaptiveFilter": lambda sig: adaptive_filtering(sig)
    }

    print("\n🧪 开始运行传统去噪方法...\n")

    # 对每个方法进行处理
    for name, method in methods.items():
        print(f"\n🔬 正在运行：{name}")
        denoised_signals = []

        for i in tqdm(range(X_test.shape[0]), desc=f"{name} Denoising"):
            denoised = method(X_test[i])
            denoised_signals.append(denoised)

        denoised_signals = np.array(denoised_signals)

        # 计算各项指标
        metrics = {
            "MSE": calculate_mse(y_test, denoised_signals),
            "MAE": calculate_mae(y_test, denoised_signals),
            "SNR": calculate_snr(y_test, denoised_signals),
            "SNR Improvement": calculate_snr_improvement(y_test, X_test, denoised_signals),
            "Pearson": calculate_pearson(y_test, denoised_signals),
            "Waveform_Fidelity": waveform_fidelity(y_test, denoised_signals),
            "SSIM": calculate_ssim_score(y_test, denoised_signals)
        }

        results[name] = {k: float(v) for k, v in metrics.items()}

        # 打印结果
        print(f"\n📊 {name} 性能指标:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        # 绘图示例
        plot_signals(y_test, X_test, denoised_signals, name, idx=0)

    # 保存到 JSON 文件
    metrics_path = os.path.join(result_dir, "traditional_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ 指标已保存至：{metrics_path}")


# ========== 辅助可视化函数 ==========

def plot_signals(original, noisy, denoised, name, idx=0):  # 添加 name 参数
    plt.figure(figsize=(14, 4))
    plt.plot(original[idx], label='Clean Signal')
    plt.plot(noisy[idx], label='Noisy Signal')
    plt.plot(denoised[idx], label='Denoised Signal')
    plt.title(f"Signal Comparison (Index {idx}) - {name}")  # 现在 name 已定义
    plt.xlabel("Time Points")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    run_experiment()
