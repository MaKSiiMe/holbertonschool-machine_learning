#!/usr/bin/env python3
"""Compatibility patches to make keras-rl2 work with TensorFlow 2.15."""

import tensorflow as tf
import tensorflow.keras.models as _km
from keras.src.saving import serialization_lib as _sl

if not hasattr(tf.keras, "__version__"):
    tf.keras.__version__ = tf.__version__

if not hasattr(_km, "model_from_config"):
    from tensorflow.keras.models import model_from_json
    _km.model_from_config = model_from_json

if hasattr(_sl, "enable_unsafe_deserialization"):
    _sl.enable_unsafe_deserialization()
