# evaluate_models.py
# 功能：加载训练好的 .h5 模型文件，重新评估并输出多种性能指标

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt
import math
from train_EOG import BasicBlockall



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

    # 只保留与噪声数据等长的 EEG 数据
    EEG_all_random = EEG_all_random[0:noise_all_random.shape[0]]

    timepoint = noise_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0])
    validation_num = round((EEG_all_random.shape[0] - train_num) / 2)
    test_num = EEG_all_random.shape[0] - train_num - validation_num

    train_eeg = EEG_all_random[0:train_num, :]
    validation_eeg = EEG_all_random[train_num:train_num + validation_num, :]
    test_eeg = EEG_all_random[train_num + validation_num:EEG_all_random.shape[0], :]

    train_noise = noise_all_random[0:train_num, :]
    validation_noise = noise_all_random[train_num:train_num + validation_num, :]
    test_noise = noise_all_random[train_num + validation_num:noise_all_random.shape[0], :]

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


# ========== 性能指标计算函数 ==========

def calculate_mse(clean, denoised):
    """均方误差（MSE）"""
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)
    return np.mean((clean - denoised) ** 2)


def calculate_mae(clean, denoised):
    """平均绝对误差（MAE）"""
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)
    return np.mean(np.abs(clean - denoised))


def calculate_snr(clean, denoised):
    """信噪比（SNR）"""
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)
    noise = clean - denoised
    snr = 10 * np.log10(np.var(clean) / np.var(noise))
    return snr


def calculate_snr_improvement(clean, noisy, denoised):
    """去噪前后 SNR 提升"""
    input_snr = calculate_snr(clean, noisy)
    output_snr = calculate_snr(clean, denoised)
    return output_snr - input_snr


def calculate_pearson(clean, denoised):
    """皮尔逊相关系数（波形一致性）"""
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
    """波形保真度（使用相关系数）"""
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
    """结构相似性指标（SSIM），适用于一维信号"""
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

if __name__ == "__main__":
    all_predictions = {}  # 新增：用于保存每个模型的预测结果

    # 路径设置
    result_dir = "results_EOG"
    file_location = r"D:\EEGdenoiseNet-master\EEGdenoiseNet-master\data\\"

    # 加载干净和带噪声的测试数据
    EEG_all = np.load(file_location + "EEG_all_epochs.npy")
    noise_all = np.load(file_location + "EOG_all_epochs.npy")

    # 准备测试集（标准化后的噪声 EEG 和 Clean EEG）
    X_test, y_test = prepare_data(EEG_all, noise_all, combin_num=10, train_per=0.8)

    # 扩展维度以适配 CNN 输入 [batch, time] -> [batch, time, 1]
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    # 模型路径
    model_paths = {
        "fcNN": os.path.join(result_dir, "fcNN_best.h5"),
        "Simple_CNN": os.path.join(result_dir, "Simple_CNN_best.h5"),
        "Complex_CNN": os.path.join(result_dir, "Complex_CNN_best.h5"),
    }

    all_metrics = {}

    # 逐个加载模型并评估
    for name, path in model_paths.items():
        print(f"\n🔄 加载并评估模型: {name}")
        # 注册自定义层
        from tensorflow.keras.utils import get_custom_objects

        get_custom_objects().update({'BasicBlockall': BasicBlockall})  # 替换为你实际的类名
        model = load_model(path, compile=False)

        # 获取预测结果
        pred = model.predict(X_test_cnn)  # 输出为 [batch, time, 1]
        pred = np.squeeze(pred)  # 去掉 channel 维度 [batch, time]
        all_predictions[name] = pred  # 保存当前模型的预测结果

        # 计算各项指标
        metrics = {
            "MSE": calculate_mse(y_test, pred),
            "MAE": calculate_mae(y_test, pred),
            "SNR": calculate_snr(y_test, pred),
            "SNR Improvement": calculate_snr_improvement(y_test, X_test, pred),  # 需要原始噪声 EEG
            "Pearson": calculate_pearson(y_test, pred),
            "Waveform_Fidelity": waveform_fidelity(y_test, pred),
            "SSIM": calculate_ssim_score(y_test, pred)
        }

        all_metrics[name] = metrics

        # 打印结果
        print(f"{name} 性能指标:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # 可视化一个样本的结果
    def plot_comparison(original, noisy, denoised, idx=0):
        plt.figure(figsize=(14, 4))
        plt.plot(original[idx], label='Clean EEG')
        plt.plot(noisy[idx], label='Noisy EEG')
        plt.plot(denoised[idx], label='Denoised EEG')
        plt.title(f"Signal Comparison (Index {idx}) - {name}")
        plt.xlabel("Time Points")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 示例可视化
    for name, res in all_metrics.items():
        idx = 0  # 可以更换索引查看不同样本
        pred = all_predictions[name]  # 取出该模型自己的预测结果
        plot_comparison(y_test, X_test, pred, idx = idx)

    # 保存指标到 JSON 文件
    import json
    with open(os.path.join(result_dir, "final_metrics.json"), "w") as f:
        json.dump({k: {mk: float(mv) for mk, mv in v.items()} for k, v in all_metrics.items()}, f, indent=4)

    print("\n📊 指标已保存至 final_metrics.json")
