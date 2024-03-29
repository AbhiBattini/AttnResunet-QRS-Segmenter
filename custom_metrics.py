from abc import ABC
import keras
import numpy as np
import tensorflow as tf


def calculate_iou(y_true, y_pred):
    y_true_binary = y_true
    y_pred_binary = tf.cast(tf.greater(y_pred, 0.7), tf.float32)
    intersection = tf.reduce_sum(tf.cast(tf.math.logical_and(tf.math.greater(y_true_binary, 0), tf.math.greater(
        y_pred_binary, 0)), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.math.logical_or(tf.math.greater(y_true_binary, 0),
                                                     tf.math.greater(y_pred_binary, 0)),
                                  tf.float32))
    iou = intersection / union
    return iou


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule, ABC):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


def DiceLoss(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_l = 1. - dice
    return dice_l


def accuracy(y_true, y_pred):
    output_tensor = tf.cast(tf.greater(y_pred, 0.7), tf.float32)
    input_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
    correct_predictions = tf.reduce_sum(tf.cast(tf.equal(output_tensor, input_tensor), dtype=tf.float32))
    total_samples = tf.cast(tf.size(output_tensor), dtype=tf.float32)
    accuracy = correct_predictions / total_samples
    return accuracy
