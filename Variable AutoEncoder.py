import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import cv2

# VAE Parameters
input_shape = (64, 64, 3)  # Adjust for your image dimensions
batch_size = 32
latent_dim = 2  # Dimension of latent space for t-SNE visualization
epochs = 50

# Encoder
def build_encoder(input_shape, latent_dim):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(32, kernel_size=3, activation='relu', strides=2, padding='same')(inputs)
    x = Conv2D(64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder

# Decoder
def build_decoder(latent_dim, original_shape):
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(16, activation='relu')(latent_inputs)
    x = Dense(16 * 16 * 64, activation='relu')(x)
    x = Reshape((16, 16, 64))(x)
    x = Conv2DTranspose(64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(3, kernel_size=3, activation='sigmoid', padding='same')(x)
    
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder

# VAE Model
encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape)

# VAE Loss
inputs = Input(shape=input_shape, name='vae_input')
z_mean, z_log_var, z = encoder(inputs)
outputs = decoder(z)
vae = Model(inputs, outputs, name='vae')

# Loss function
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs)) * np.prod(input_shape)
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1) * -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Load and preprocess data (assuming you have a dataset)
# Example: dataset of images and class labels
# Replace with your actual data loading function
# images, labels = load_your_dataset()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the VAE
vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None))

# Latent space encoding and t-SNE visualization
z_mean, _, _ = encoder.predict(X_test, batch_size=batch_size)
tsne = TSNE(n_components=2, random_state=42)
z_tsne = tsne.fit_transform(z_mean)

# Plot t-SNE
plt.figure(figsize=(10, 8))
sns.scatterplot(x=z_tsne[:, 0], y=z_tsne[:, 1], hue=y_test, palette='viridis', s=60)
plt.title('t-SNE Visualization of Latent Space')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar()
plt.show()

# Histogram of classes (invasive species)
plt.figure(figsize=(8, 6))
sns.histplot(y_train, kde=False, bins=len(np.unique(y_train)))
plt.title("Histogram of Invasive Species Classes")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Train-test split pie chart
train_pct = len(X_train) / (len(X_train) + len(X_test)) * 100
test_pct = 100 - train_pct
plt.figure(figsize=(6, 6))
plt.pie([train_pct, test_pct], labels=['Training Data', 'Testing Data'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title("Training and Testing Data Split")
plt.show()

# Display an original and reconstructed image
sample_img = X_test[0]
reconstructed_img = vae.predict(np.expand_dims(sample_img, axis=0))[0]

# Clip and display images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(np.clip(sample_img, 0, 1))  # Clipping to [0,1] range for RGB
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(np.clip(reconstructed_img, 0, 1))  # Clipping to [0,1] range for RGB
plt.title("Reconstructed Image")
plt.show()
