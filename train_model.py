import os
import pickle
import ecgmentations as E
import keras
import numpy as np
import scipy
import wfdb
from matplotlib import pyplot as plt
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from custom_metrics import DiceLoss, accuracy, calculate_iou, WarmUpCosine
from newqrsmodel import unet

transform = E.Sequential([
    E.GaussNoise(variance=0.0005),
    E.GaussBlur(),
    E.AmplitudeScale(),
    E.PowerlineNoise(ecg_frequency=250),
    E.PowerlineNoise(ecg_frequency=250, always_apply=True),
    E.SinePulse(ecg_frequency=250),
    E.RespirationNoise(ecg_frequency=250, always_apply=True),
    E.Blur(p=0.1)
])


def bandpass_filter(r_sig, sampling_rate, low_cutoff, high_cutoff, order=5):
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    ba, ab = scipy.signal.butter(order, [low, high], btype='band')
    filtered_sig = scipy.signal.filtfilt(ba, ab, r_sig)
    return filtered_sig


def clean(ecg_signal, frequency):
    newst = bandpass_filter(np.squeeze(ecg_signal), frequency, 0.5, 45)
    return newst


def point(record_sig, qrs_loc):
    global ori, end
    ori = qrs_loc[0] - 150
    end = qrs_loc[-1] + 150
    if ori < 0:
        ori = 0
    if end > len(record_sig.squeeze()) - 1:
        end = len(record_sig.squeeze()) - 1
    return ori, end


def create_truth(locations, symbol, signal, start):
    qrs_comp = np.zeros(2000)
    length = len(np.squeeze(signal))
    qrs_d = {}
    for i, i_sample in enumerate(locations):
        qrs_d[i] = i_sample
    qrs_values = list(qrs_d.values())
    for z, z_s in enumerate(locations):
        if symbol[z] == 'N':
            index = qrs_values.index(z_s)
            if symbol[index - 1] == "(":
                left = int((qrs_values[index - 1] - start) * (2000 / length))
            else:
                left = int((qrs_values[index] - start) * (2000 / length))
            try:
                if symbol[index + 1] == ")":
                    right = int((qrs_values[index + 1] - start) * (2000 / length))
                else:
                    right = int((qrs_values[index] - start) * (2000 / length))
            except IndexError:
                right = int((qrs_values[index] - start) * (2000 / length))
            qrs_comp[left:right] = 1
    return np.expand_dims(qrs_comp, axis=-1)


def collect(file, inputting, true):
    print(file)
    record1 = wfdb.rdrecord(file, channels=[0], warn_empty=True, pn_dir='ludb/data')
    record2 = wfdb.rdrecord(file, channels=[1], warn_empty=True, pn_dir='ludb/data')
    record3 = wfdb.rdrecord(file, channels=[2], warn_empty=True, pn_dir='ludb/data')
    record4 = wfdb.rdrecord(file, channels=[3], warn_empty=True, pn_dir='ludb/data')
    record5 = wfdb.rdrecord(file, channels=[4], warn_empty=True, pn_dir='ludb/data')
    record6 = wfdb.rdrecord(file, channels=[5], warn_empty=True, pn_dir='ludb/data')
    record7 = wfdb.rdrecord(file, channels=[6], warn_empty=True, pn_dir='ludb/data')
    record8 = wfdb.rdrecord(file, channels=[7], warn_empty=True, pn_dir='ludb/data')
    record9 = wfdb.rdrecord(file, channels=[8], warn_empty=True, pn_dir='ludb/data')
    record10 = wfdb.rdrecord(file, channels=[9], warn_empty=True, pn_dir='ludb/data')
    record11 = wfdb.rdrecord(file, channels=[10], warn_empty=True, pn_dir='ludb/data')
    record12 = wfdb.rdrecord(file, channels=[11], warn_empty=True, pn_dir='ludb/data')
    i = wfdb.rdann(file, 'i', pn_dir='ludb/data')
    ii = wfdb.rdann(file, 'ii', pn_dir='ludb/data')
    iii = wfdb.rdann(file, 'iii', pn_dir='ludb/data')
    avr = wfdb.rdann(file, 'avr', pn_dir='ludb/data')
    avl = wfdb.rdann(file, 'avl', pn_dir='ludb/data')
    avf = wfdb.rdann(file, 'avf', pn_dir='ludb/data')
    v1 = wfdb.rdann(file, 'v1', pn_dir='ludb/data')
    v2 = wfdb.rdann(file, 'v2', pn_dir='ludb/data')
    v3 = wfdb.rdann(file, 'v3', pn_dir='ludb/data')
    v4 = wfdb.rdann(file, 'v4', pn_dir='ludb/data')
    v5 = wfdb.rdann(file, 'v5', pn_dir='ludb/data')
    v6 = wfdb.rdann(file, 'v6', pn_dir='ludb/data')
    i_locations = i.sample
    ii_locations = ii.sample
    iii_locations = iii.sample
    avr_locations = avr.sample
    avl_locations = avl.sample
    avf_locations = avf.sample
    v1_locations = v1.sample
    v2_locations = v2.sample
    v3_locations = v3.sample
    v4_locations = v4.sample
    v5_locations = v5.sample
    v6_locations = v6.sample
    i_sym = i.symbol
    ii_sym = ii.symbol
    iii_sym = iii.symbol
    avr_sym = avr.symbol
    avl_sym = avl.symbol
    avf_sym = avf.symbol
    v1_sym = v1.symbol
    v2_sym = v2.symbol
    v3_sym = v3.symbol
    v4_sym = v4.symbol
    v5_sym = v5.symbol
    v6_sym = v6.symbol
    clean1 = record1.p_signal
    clean2 = record2.p_signal
    clean3 = record3.p_signal
    clean4 = record4.p_signal
    clean5 = record5.p_signal
    clean6 = record6.p_signal
    clean7 = record7.p_signal
    clean8 = record8.p_signal
    clean9 = record9.p_signal
    clean10 = record10.p_signal
    clean11 = record11.p_signal
    clean12 = record12.p_signal
    l1, r1 = point(clean1, i_locations)
    l2, r2 = point(clean2, ii_locations)
    l3, r3 = point(clean3, iii_locations)
    l4, r4 = point(clean4, avr_locations)
    l5, r5 = point(clean5, avl_locations)
    l6, r6 = point(clean6, avf_locations)
    l7, r7 = point(clean7, v1_locations)
    l8, r8 = point(clean8, v2_locations)
    l9, r9 = point(clean9, v3_locations)
    l10, r10 = point(clean10, v4_locations)
    l11, r11 = point(clean11, v5_locations)
    l12, r12 = point(clean12, v6_locations)
    true_array1 = create_truth(i_locations, i_sym, clean1[l1:r1], l1)
    true_array2 = create_truth(ii_locations, ii_sym, clean2[l2:r2], l2)
    true_array3 = create_truth(iii_locations, iii_sym, clean3[l3:r3], l3)
    true_array4 = create_truth(avr_locations, avr_sym, clean4[l4:r4], l4)
    true_array5 = create_truth(avl_locations, avl_sym, clean5[l5:r5], l5)
    true_array6 = create_truth(avf_locations, avf_sym, clean6[l6:r6], l6)
    true_array7 = create_truth(v1_locations, v1_sym, clean7[l7:r7], l7)
    true_array8 = create_truth(v2_locations, v2_sym, clean8[l8:r8], l8)
    true_array9 = create_truth(v3_locations, v3_sym, clean9[l9:r9], l9)
    true_array10 = create_truth(v4_locations, v4_sym, clean10[l10:r10], l10)
    true_array11 = create_truth(v5_locations, v5_sym, clean11[l11:r11], l11)
    true_array12 = create_truth(v6_locations, v6_sym, clean12[l12:r12], l12)
    new_rec1 = np.expand_dims(resample(clean1.squeeze()[l1:r1], 2000), axis=1)
    new_rec2 = np.expand_dims(resample(clean2.squeeze()[l2:r2], 2000), axis=1)
    new_rec3 = np.expand_dims(resample(clean3.squeeze()[l3:r3], 2000), axis=1)
    new_rec4 = np.expand_dims(resample(clean4.squeeze()[l4:r4], 2000), axis=1)
    new_rec5 = np.expand_dims(resample(clean5.squeeze()[l5:r5], 2000), axis=1)
    new_rec6 = np.expand_dims(resample(clean6.squeeze()[l6:r6], 2000), axis=1)
    new_rec7 = np.expand_dims(resample(clean7.squeeze()[l7:r7], 2000), axis=1)
    new_rec8 = np.expand_dims(resample(clean8.squeeze()[l8:r8], 2000), axis=1)
    new_rec9 = np.expand_dims(resample(clean9.squeeze()[l9:r9], 2000), axis=1)
    new_rec10 = np.expand_dims(resample(clean10.squeeze()[l10:r10], 2000), axis=1)
    new_rec11 = np.expand_dims(resample(clean11.squeeze()[l11:r11], 2000), axis=1)
    new_rec12 = np.expand_dims(resample(clean12.squeeze()[l12:r12], 2000), axis=1)
    inputting.append(new_rec1), inputting.append(new_rec2), inputting.append(new_rec3)
    inputting.append(new_rec4), inputting.append(new_rec5), inputting.append(new_rec6)
    inputting.append(new_rec7), inputting.append(new_rec8), inputting.append(new_rec9)
    inputting.append(new_rec10), inputting.append(new_rec11), inputting.append(new_rec12)
    true.append(true_array1), true.append(true_array2), true.append(true_array3)
    true.append(true_array4), true.append(true_array5), true.append(true_array6)
    true.append(true_array7), true.append(true_array8), true.append(true_array9)
    true.append(true_array10), true.append(true_array11), true.append(true_array12)



q = wfdb.get_record_list('ludb')
w = []
for name in q:
    if len(name) == 6:
        w.append(name[-1:])
    if len(name) == 7:
        w.append(name[-2:])
    if len(name) == 8:
        w.append(name[-3:])
w.remove('8'), w.remove('111')
all_inputs = []
all_true = []
for subject in w:
    try:
        collect(subject, all_inputs, all_true)
    except KeyboardInterrupt:
        plt.plot(all_inputs[-1][:, :])
        plt.plot(np.squeeze(all_true[-1])[:])
        plt.show()

input_signals = np.concatenate([np.expand_dims(np.array(transform(ecg=np.squeeze(np.array(all_inputs)))['ecg']),
                                               axis=-1),
                                np.array(all_inputs)])
truth_signals = np.concatenate([np.array(all_true), np.array(all_true)], axis=0)
X_train, X_val, y_train, y_val = train_test_split(input_signals, truth_signals, test_size=0.1, random_state=42)
print(X_train.shape)
model = unet((None, 1))
WEIGHT_DECAY = 5e-4
epochs = 20
steps = 173
lr_decayed_fn = WarmUpCosine(  # Change with BT_UNET
    learning_rate_base=6e-3,
    total_steps=epochs * steps,
    warmup_learning_rate=0.0,
    warmup_steps=steps
)
optimizer = keras.optimizers.Adam(learning_rate=lr_decayed_fn)
model.compile(optimizer=optimizer, loss=DiceLoss, metrics=[accuracy, keras.metrics.Precision(),
                                                           keras.metrics.Recall(),
                                                           keras.metrics.AUC(curve='ROC'),
                                                           calculate_iou])
hist = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=25)
model.save_weights('QRSUNETf1.h5')
