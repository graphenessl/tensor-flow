### Speech recognition with Tensor flow

# Requirements
- scikit-image
- tflearn
- tensorflow
- librosa
- enum
- h5py

You can install those with `pip` or your preferred Python package manager

# Description
This is an end-to-end implementation of voice speech recognition using Python and TFLearn, which is a high-level
API for TensorFlow. The model downloads a sample training list from the Internet, but you can also record your own one.

# Trained model
Existing trained model available at: `0-10_digits_model.tflearn`
You can use this model to do predictions. However, note that this model has been trained only with 100 iterations.
You can train your own one with more steps(iterations), like 100 000, for better accuracy. Or play with the
`learning_rate` in order to find your ideal time vs accuracy ratio.

# How to use
Place your .wav file containing a pronounciation of a digit [0-9] in the root directory of the project and run:
`python speech_recognize.py your-file-containing-digit-sound.wav`
