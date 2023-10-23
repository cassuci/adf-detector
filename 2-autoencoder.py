import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import pandas as pd
#import librosa
import logging
import os   
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# In[2]:


tf.config.list_physical_devices('GPU')


# In[ ]:

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def audio_to_mel_spectrogram(audio_file_path, max_length=4*22500, sr=22500, n_mels=256):
    audio_file_path = audio_file_path if type(audio_file_path) == str else audio_file_path.decode('ASCII')
    audio_file_path = str(audio_file_path).replace('\\\\', os.sep)
    audio_file_path = audio_file_path.replace('F:', '/mnt/f').replace('//', '/')
    
    with open(audio_file_path, 'rb') as fh:
        y, sr = librosa.load(fh, sr=sr)
    
    if len(y) > max_length:
        # If the audio is longer than the maximum length, cut it
        y = y[:max_length]
    elif len(y) < max_length:
        # If the audio is shorter, pad it with zeros
        pad_length = max_length - len(y)
        y = np.pad(y, (0, pad_length), mode='constant')
    
    # Calculate the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram


# In[ ]:


# Generator function for lazy loading of audio data
def audio_data_generator(audio_files):
    for audio_file in audio_files:
        sample = (audio_to_mel_spectrogram(audio_file) + 80)/80
        yield sample.reshape((1, 128, 176, 1)), sample.reshape((1, 128, 176, 1))


# In[ ]:


vae_train = pd.read_csv('vae_train.csv')
vae_train_generator = tf.data.Dataset.from_generator(audio_data_generator, args=[vae_train['full_path']], output_signature=(tf.TensorSpec(shape=(1, 128, 176, 1)),tf.TensorSpec(shape=(1, 128, 176, 1))))


# In[ ]:


vae_val = pd.read_csv('vae_val.csv')
vae_val_generator = tf.data.Dataset.from_generator(audio_data_generator, args=[vae_val['full_path']], output_signature=(tf.TensorSpec(shape=(1, 128, 176, 1)),tf.TensorSpec(shape=(1, 128, 176, 1))))


# In[ ]:


vae_test = pd.read_csv('vae_test.csv')
vae_test_generator = tf.data.Dataset.from_generator(audio_data_generator, args=[vae_test['full_path']], output_signature=(tf.TensorSpec(shape=(1, 128, 176, 1)),tf.TensorSpec(shape=(1, 128, 176, 1))))


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Normalization, Reshape, Dense, Flatten, Cropping2D
from tensorflow.keras.models import Model

# Define the input shape
input_shape = (128, 176, 1)  # You can adjust the input shape based on your data

# Define the encoder
input_layer = Input(shape=input_shape)
x = input_layer

x = Conv2D(256, kernel_size = (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dense(512)(x)
encoded = x


x = Dense(8*11*16)(x)
x = Reshape((8, 11, 16))(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
#x = Cropping2D(cropping=((3, 0), (0, 0)))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_layer, decoded)

# Compile the autoencoder
mse = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.SUM)

autoencoder.compile(optimizer='adam', loss=mse, metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

# Print the model summary
autoencoder.summary()


# In[ ]:


autoencoder = tf.keras.saving.load_model('autoencoder_model')
checkpoint_path = 'autoencoder.ckpt'
autoencoder.load_weights(checkpoint_path)



# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

logging.info('Training the autoencoder...')
# Create the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=4096)


# Train the model with class weights and early stopping
history = autoencoder.fit(vae_train_generator, epochs=300, batch_size=512, validation_data=vae_val_generator, callbacks=[early_stopping, cp_callback])

logging.info('Training complete.')


# In[ ]:


#train 3,243 batches of 128 (415k)
#val 1,298 batches of 128 (166k)


# In[ ]:


autoencoder.save('autoencoder_model')


# In[ ]:


import soundfile as sf  # Soundfile library for saving the WAV file
def mel_spectrogram_to_audio(mel_spectrogram, max_length=4*22500, sr=22500):
    # Inverse of power_to_db
    mel_spectrogram = librosa.db_to_power(mel_spectrogram)
    # Inverse of melspectrogram
    y = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr)
    
    return y

# Example usage
spectrogram = audio_to_mel_spectrogram("converted.wav", sr=44100).reshape((1, 128, 176, 1))
out_spectre = autoencoder.predict(spectrogram)
out_spectre = (out_spectre*80)-80
out_spectre = out_spectre.reshape((128, 176))
reconstructed_audio = mel_spectrogram_to_audio(out_spectre, 44100) * 1e2
sf.write("reconstructed_audio.wav", np.ravel(reconstructed_audio), 44100)  # Adjust the sample rate as needed


# In[ ]:


y, sr = librosa.load('converted.wav', sr=22500, duration=112500)
plt.plot(y)


# In[ ]:


plt.plot(reconstructed_audio)


# In[ ]:


plt.imshow(out_spectre)


# In[ ]:


plt.imshow(spectrogram[0])


# In[ ]:


import soundfile as sf  # Soundfile library for saving the WAV file
def mel_spectrogram_to_audio(mel_spectrogram, max_length=4*22500, sr=22500):
    # Inverse of power_to_db
    mel_spectrogram = librosa.db_to_power(mel_spectrogram)
    # Inverse of melspectrogram
    y = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr)
    
    return y

sample = next(vae_train_generator.as_numpy_iterator())[0]
# Example usage
out_spectre = autoencoder.predict(sample)
out_spectre = (out_spectre*80)-80
out_spectre = out_spectre.reshape((128, 176))
plt.imshow(sample.reshape((128, 176)))


# In[ ]:


plt.imshow(out_spectre.reshape((128, 176)))


# In[ ]:


a = spectrogram.reshape((128, 176))
b = out_spectre
(((a+80)/80-(b+80)/80)**2).mean(),\
(((a+80)/80-(b+80)/80)**2).sum()


# In[ ]:




