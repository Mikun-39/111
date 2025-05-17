# train_EOG.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

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

    EEG_train = random_signal(signal=train_eeg, combin_num=combin_num).reshape(combin_num * train_eeg.shape[0], timepoint)
    NOISE_train = random_signal(signal=train_noise, combin_num=combin_num).reshape(combin_num * train_noise.shape[0], timepoint)

    SNR_train_dB = np.random.uniform(-7, 2, (EEG_train.shape[0]))
    SNR_train = 10 ** (0.1 * (SNR_train_dB))

    noiseEEG_train = []
    NOISE_train_adjust = []
    for i in range(EEG_train.shape[0]):
        eeg = EEG_train[i].reshape(EEG_train.shape[1])
        noise = NOISE_train[i].reshape(NOISE_train.shape[1])
        coe = get_rms(eeg) / (get_rms(noise) * SNR_train[i])
        noise = noise * coe
        neeg = noise + eeg
        NOISE_train_adjust.append(noise)
        noiseEEG_train.append(neeg)

    noiseEEG_train = np.array(noiseEEG_train)
    NOISE_train_adjust = np.array(NOISE_train_adjust)

    EEG_train_end_standard = noiseEEG_train / np.std(noiseEEG_train, axis=1, keepdims=True)
    noiseEEG_train_end_standard = noiseEEG_train / np.std(noiseEEG_train, axis=1, keepdims=True)

    # 验证集构造
    SNR_val_dB = np.linspace(-7.0, 2.0, num=10)
    SNR_val = 10 ** (0.1 * (SNR_val_dB))

    EEG_val = []
    noise_EEG_val = []
    for i in range(10):
        noise_eeg_val = []
        for j in range(validation_eeg.shape[0]):
            eeg = validation_eeg[j]
            noise = validation_noise[j]
            coe = get_rms(eeg) / (get_rms(noise) * SNR_val[i])
            noise = noise * coe
            neeg = noise + eeg
            noise_eeg_val.append(neeg)
        EEG_val.extend(validation_eeg)
        noise_EEG_val.extend(noise_eeg_val)

    noise_EEG_val = np.array(noise_EEG_val)
    EEG_val = np.array(EEG_val)

    EEG_val_end_standard = EEG_val / np.std(noise_EEG_val, axis=1, keepdims=True)
    noiseEEG_val_end_standard = noise_EEG_val / np.std(noise_EEG_val, axis=1, keepdims=True)

    # 测试集构造
    SNR_test_dB = np.linspace(-7.0, 2.0, num=10)
    SNR_test = 10 ** (0.1 * (SNR_test_dB))

    EEG_test = []
    noise_EEG_test = []
    for i in range(10):
        noise_eeg_test = []
        for j in range(test_eeg.shape[0]):
            eeg = test_eeg[j]
            noise = test_noise[j]
            coe = get_rms(eeg) / (get_rms(noise) * SNR_test[i])
            noise = noise * coe
            neeg = noise + eeg
            noise_eeg_test.append(neeg)
        EEG_test.extend(test_eeg)
        noise_EEG_test.extend(noise_eeg_test)

    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)

    EEG_test_end_standard = EEG_test / np.std(noise_EEG_test, axis=1, keepdims=True)
    noiseEEG_test_end_standard = noise_EEG_test / np.std(noise_EEG_test, axis=1, keepdims=True)

    return (
        noiseEEG_train_end_standard,
        EEG_train_end_standard,
        noiseEEG_val_end_standard,
        EEG_val_end_standard,
        noiseEEG_test_end_standard,
        EEG_test_end_standard,
    )

# ========== 损失函数 ==========
def denoise_loss_mse(denoise, clean):
    if len(denoise.shape) == 3:
        denoise = tf.squeeze(denoise, axis=-1)
    if len(clean.shape) == 3:
        clean = tf.squeeze(clean, axis=-1)
    loss = tf.losses.mean_squared_error(denoise, clean)
    return tf.reduce_mean(loss)

# ========== 网络结构 ==========
def fcNN(datanum):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(datanum,)))
    model.add(layers.Dense(datanum, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(datanum, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(datanum, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(datanum))
    model.summary()
    return model

def simple_CNN(datanum):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, 3, strides=1, padding='same', input_shape=[datanum, 1]))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(datanum))
    model.build(input_shape=[1, datanum, 1])
    model.summary()
    return model

def Complex_CNN(datanum):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(32, 5, strides=1, padding="same", input_shape=[datanum, 1]))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(BasicBlockall())
    model.add(layers.Conv1D(32, 1, strides=1, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(datanum))
    model.build(input_shape=[1, datanum, 1])
    model.summary()
    return model

class Res_BasicBlock(layers.Layer):
    def __init__(self, kernelsize, stride=1):
        super(Res_BasicBlock, self).__init__()
        self.bblock = tf.keras.Sequential([
            layers.Conv1D(32, kernelsize, strides=stride, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(16, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(32, kernelsize, strides=1, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.jump_layer = lambda x: x

    def call(self, inputs, training=None):
        out = self.bblock(inputs)
        identity = self.jump_layer(inputs)
        return layers.add([out, identity])

class BasicBlockall(layers.Layer):
    def __init__(self, stride=1):
        super(BasicBlockall, self).__init__()
        self.bblock3 = tf.keras.Sequential([Res_BasicBlock(3), Res_BasicBlock(3)])
        self.bblock5 = tf.keras.Sequential([Res_BasicBlock(5), Res_BasicBlock(5)])
        self.bblock7 = tf.keras.Sequential([Res_BasicBlock(7), Res_BasicBlock(7)])

    def call(self, inputs, training=None):
        out3 = self.bblock3(inputs)
        out5 = self.bblock5(inputs)
        out7 = self.bblock7(inputs)
        return tf.concat([out3, out5, out7], axis=-1)

# ========== 训练方法 ==========
def train_step(model, inputs, labels, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        if len(predictions.shape) == 3:
            predictions = tf.squeeze(predictions, axis=-1)
        if len(labels.shape) == 3:
            labels = tf.squeeze(labels, axis=-1)
        loss = denoise_loss_mse(predictions, labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def evaluate_model(model, noiseEEG_test, EEG_test):
    denoiseoutput_test = model(noiseEEG_test)
    if len(denoiseoutput_test.shape) == 3:
        denoiseoutput_test = tf.squeeze(denoiseoutput_test, axis=-1)
    if len(EEG_test.shape) == 2:
        EEG_test = tf.expand_dims(EEG_test, axis=-1)
    loss = denoise_loss_mse(EEG_test, denoiseoutput_test)
    return denoiseoutput_test, loss

def train_model(model, name, noiseEEG, EEG, noiseEEG_val, EEG_val, epochs, batch_size, result_dir):
    history = {'train': [], 'val': []}
    best_loss = float('inf')
    batch_num = math.ceil(noiseEEG.shape[0] / batch_size)
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)

    print(f"\n🚀 Training {name}...")
    for epoch in range(epochs):
        start = time.time()
        train_loss = 0
        with tqdm(total=batch_num, position=0, leave=True) as pbar:
            for n_batch in range(batch_num):
                start_idx = n_batch * batch_size
                end_idx = min((n_batch + 1) * batch_size, noiseEEG.shape[0])
                noiseEEG_batch = tf.convert_to_tensor(noiseEEG[start_idx:end_idx], dtype=tf.float32)
                EEG_batch = tf.convert_to_tensor(EEG[start_idx:end_idx], dtype=tf.float32)
                loss = train_step(model, noiseEEG_batch, EEG_batch, optimizer)
                train_loss += loss.numpy()
                pbar.update()
            pbar.close()

        avg_train_loss = train_loss / batch_num
        val_pred, val_loss = evaluate_model(model, noiseEEG_val, EEG_val)
        val_loss = val_loss.numpy()

        history['train'].append(avg_train_loss)
        history['val'].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time() - start:.2f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            model.save(os.path.join(result_dir, f"{name}_best.h5"))

    return model, history

# ========== 可视化函数 ==========
def plot_losses(history_dict):
    plt.figure(figsize=(12, 6))
    for name, h in history_dict.items():
        plt.plot(h['train'], label=f'{name} Train Loss')
        plt.plot(h['val'], '--', label=f'{name} Val Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_signals(original, noisy, denoised, idx=0):
    plt.figure(figsize=(14, 4))
    plt.plot(original[idx], label='Clean EEG')
    plt.plot(noisy[idx], label='Noisy EEG')
    plt.plot(denoised[idx], label='Denoised EEG')
    plt.title(f"Signal Comparison (Index {idx})")
    plt.xlabel("Time Points")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_snr(clean, denoised):
    if len(clean.shape) == 3:
        clean = np.squeeze(clean, axis=-1)
    if len(denoised.shape) == 3:
        denoised = np.squeeze(denoised, axis=-1)
    noise = clean - denoised
    snr = 10 * np.log10(np.var(clean) / np.var(noise))
    return snr

# ========== 主程序 ==========
if __name__ == "__main__":
    # 参数设置
    epochs = 20
    batch_size = 32
    data_ratio = 1
    result_dir = "results_EOG"
    os.makedirs(result_dir, exist_ok=True)

    # 加载数据
    file_location = r"D:\EEGdenoiseNet-master\EEGdenoiseNet-master\data\\"
    EEG_all = np.load(file_location + "EEG_all_epochs.npy")
    noise_all = np.load(file_location + "EOG_all_epochs.npy")
    datanum = 512

    # 准备训练/验证/测试数据
    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test = prepare_data(
        EEG_all, noise_all, combin_num=10, train_per=0.8
    )

    # 取部分数据用于快速测试
    train_size = int(noiseEEG_train.shape[0] * data_ratio)
    noiseEEG_train = noiseEEG_train[:train_size]
    EEG_train = EEG_train[:train_size]

    # 存储结果
    results = {}
    model_builders = {
        "fcNN": fcNN,
        "Simple_CNN": simple_CNN,
        "Complex_CNN": Complex_CNN,
    }

    for name, builder in model_builders.items():
        model = builder(datanum)

        if len(model.input_shape) == 2 and model.input_shape[1] != noiseEEG_train.shape[1]:
            raise ValueError(f"Model {name} expects input shape {model.input_shape}, but got {noiseEEG_train.shape[1]}")

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mse')

        if name == "fcNN":
            X_train_input = noiseEEG_train
            y_train_input = EEG_train
            X_val_input = noiseEEG_val
            y_val_input = EEG_val
        else:
            X_train_input = np.expand_dims(noiseEEG_train, axis=-1)
            y_train_input = np.expand_dims(EEG_train, axis=-1)
            X_val_input = np.expand_dims(noiseEEG_val, axis=-1)
            y_val_input = np.expand_dims(EEG_val, axis=-1)

        trained_model, history = train_model(
            model, name,
            X_train_input, y_train_input,
            X_val_input, y_val_input,
            epochs=epochs, batch_size=batch_size,
            result_dir=result_dir
        )

        results[name] = {
            "model": trained_model,
            "history": history,
            "X_test": np.expand_dims(noiseEEG_test, axis=-1),
            "y_test": np.expand_dims(EEG_test, axis=-1)
        }

    # 绘制 loss 曲线
    plot_hist = {k: v["history"] for k, v in results.items()}
    plot_losses(plot_hist)

    # 评估模型并计算 SNR
    all_results = {}
    for name, res in results.items():
        print(f"\n📊 Evaluating {name} on EOG")
        pred, loss = evaluate_model(res["model"], res["X_test"], res["y_test"])
        snr = calculate_snr(res["y_test"], pred)
        print(f"Test Loss: {loss:.4f}, SNR: {snr:.2f} dB")
        all_results[f"{name}_EOG"] = {"loss": loss, "snr": snr}
        plot_signals(res["y_test"], res["X_test"], pred, idx=0)

    # 打印最终 SNR 结果
    print("\n📈 Final SNR Results:")
    for key, value in all_results.items():
        print(f"{key}: SNR={value['snr']:.2f} dB")
