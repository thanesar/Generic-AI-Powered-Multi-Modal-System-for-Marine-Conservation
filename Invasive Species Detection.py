import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# VAE Parameters
input_shape = (64, 64, 3)  # Adjust to your image dimensions
batch_size = 32
latent_dim = 2  # Dimension of the latent space
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

# Build the VAE Model
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

# Load and preprocess data (assuming you have a dataset with native species only for training)
# Example: dataset of native species images
# Replace with your actual data loading function
# native_images, invasive_images = load_your_data()

# Train-test split for native species
X_train, X_test = train_test_split(native_images, test_size=0.2, random_state=42)

# Train the VAE on native species
vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None))

# Function to calculate reconstruction error
def calculate_reconstruction_error(vae, images):
    reconstructed_images = vae.predict(images)
    errors = [mean_squared_error(image.flatten(), reconstructed.flatten()) for image, reconstructed in zip(images, reconstructed_images)]
    return errors

# Set a threshold for anomaly detection
# Calculate reconstruction error on test set (native species)
native_errors = calculate_reconstruction_error(vae, X_test)
threshold = np.mean(native_errors) + 2 * np.std(native_errors)

# Detect anomalies (invasive species) by testing error on invasive images
invasive_errors = calculate_reconstruction_error(vae, invasive_images)
native_anomalies = calculate_reconstruction_error(vae, X_test)

# Plot histogram of reconstruction errors for native and invasive species
plt.figure(figsize=(10, 6))
sns.histplot(native_errors, color="green", label="Native Species", kde=True, bins=30)
sns.histplot(invasive_errors, color="red", label="Invasive Species", kde=True, bins=30)
plt.axvline(threshold, color='blue', linestyle='--', label="Anomaly Threshold")
plt.legend()
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Histogram for Native and Invasive Species")
plt.show()

# Calculate percentage of anomalies detected in invasive species
invasive_anomalies = np.sum(np.array(invasive_errors) > threshold)
print(f"Anomaly Detection Rate for Invasive Species: {invasive_anomalies / len(invasive_images) * 100:.2f}%")

# Pie chart showing distribution of anomalies detected
anomalies_detected = np.sum(np.array(invasive_errors) > threshold)
normal_detected = len(invasive_errors) - anomalies_detected
plt.figure(figsize=(6, 6))
plt.pie([normal_detected, anomalies_detected], labels=['Non-Anomalous', 'Anomalous'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title("Anomaly Detection in Invasive Species")
plt.show()
