import pandas as pd
import gdown
import plotly.express as px
from io import StringIO
import requests
import matplotlib.pyplot as plt
import tempfile
import os
import nibabel as nib
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical




csv_url = "https://raw.githubusercontent.com/karimconf/medical_data/main/Train_sexAge.csv"

# Make a request to the URL and get the content
response = requests.get(csv_url)
content = response.text

# Use pandas to read the CSV data
df = pd.read_csv(StringIO(content))
df = df.drop('Unnamed: 4', axis=1)



github_repo_url = "https://raw.githubusercontent.com/karimconf/medical_data/main/Train/"
df['image_link'] = df['ID'].apply(lambda x: f"{github_repo_url}areg_{x}_brain.nii.gz")

count_by_sex = df[df['target'].isin(['ADHD', 'hCon'])].groupby(['target', 'Sex']).size().reset_index(name='count')


# Display the result
print(count_by_sex)
df.head()


adhd_data = df[df['target'] == 'ADHD'].sort_values(by=['Sex', 'Age'])
hcon_data = df[df['target'] == 'hCon'].sort_values(by=['Sex', 'Age'])

fig_adhd = px.bar(adhd_data, x='ID', y='Age', color='Sex', title='ADHD Data',
                  hover_data=['Sex', 'Age', 'image_link'])
fig_hcon = px.bar(hcon_data, x='ID', y='Age', color='Sex', title='hCon Data',
                  hover_data=['Sex', 'Age', 'image_link'])

# Show the plots
fig_adhd.show()
fig_hcon.show()





print("ADHD : ",adhd_data.describe())

print("hCon : ",hcon_data.describe())







nii_files = [
    "https://raw.githubusercontent.com/karimconf/medical_data/main/Train/areg_ADHD_1189_brain.nii.gz",
    "https://raw.githubusercontent.com/karimconf/medical_data/main/Train/areg_hCon_1012_brain.nii.gz"
]

# Create a temporary directory to store downloaded files
temp_dir = tempfile.mkdtemp()

# Create a single plot with two subplots
fig, axs = plt.subplots(1, 2, figsize=(6, 5))  # Adjust the figsize as needed

# Loop through each NIfTI file
for i, nii_file_url in enumerate(nii_files):
    try:
        # Download the file content
        response = requests.get(nii_file_url)
        response.raise_for_status()  # Raise an error for bad responses
        file_content = response.content

        # Save the content to a temporary file
        temp_file_path = os.path.join(temp_dir, f"temp_nii_{i}.nii.gz")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        # Load NIfTI image using nibabel
        nii_image = nib.load(temp_file_path)

        # Get the NIfTI image data as a 3D NumPy array
        image_data = nii_image.get_fdata()

        # Display the NIfTI image in a subplot
        axs[i].imshow(image_data[:, :, image_data.shape[2] // 2],
                      extent=[0, image_data.shape[1], 0, image_data.shape[0]],
                      cmap='gray', aspect='equal')
        axs[i].set_title(os.path.basename(nii_file_url))

    except Exception as e:
        print(f"Error loading image {nii_file_url}: {e}")

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

# Clean up temporary directory
for file_path in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, file_path))
os.rmdir(temp_dir)












image_data = []
labels = []
temp_dir = tempfile.mkdtemp()

try:
    for index, row in df.iterrows():
        try:
            nii_file_url = row['image_link']
            response = requests.get(nii_file_url)
            response.raise_for_status()

            # Save the NIfTI file content to a temporary file
            temp_file_path = os.path.join(temp_dir, f"temp_nii_{index}.nii.gz")
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(response.content)

            # Load NIfTI image using nibabel
            nii_image = nib.load(temp_file_path)
            nii_array = np.asarray(nii_image.get_fdata())

            image_data.append(nii_array)
            labels.append(row['target'])

        except Exception as e:
            print(f"Error loading image {nii_file_url}: {e}")

finally:
    # Clean up temporary directory
    for file_path in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file_path))
    os.rmdir(temp_dir)


# Convert labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels_categorical = to_categorical(encoded_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    np.array(image_data), encoded_labels_categorical, test_size=0.2, random_state=42
)

# Reshape data to explicitly include the depth dimension
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1))

# Define LeNet model with Conv3D layers
lenet_model = models.Sequential([
    layers.Conv3D(6, (3, 3, 3), activation='relu', input_shape=X_train.shape[1:]),
    layers.AveragePooling3D(),
    layers.Conv3D(16, (3, 3, 3), activation='relu'),
    layers.AveragePooling3D(),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(2, activation='softmax')  # Assuming 2 classes (ADHD and hCon)
])

# Compile the LeNet model
lenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the LeNet model
lenet_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the LeNet model on the test set
lenet_test_loss, lenet_test_accuracy = lenet_model.evaluate(X_test, y_test)
print(f'LeNet Test Accuracy: {lenet_test_accuracy * 100:.2f}%')

# Define Dense model
dense_model = models.Sequential([
    layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # Assuming 2 classes (ADHD and hCon)
])

# Compile the Dense model
dense_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Dense model
dense_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the Dense model on the test set
dense_test_loss, dense_test_accuracy = dense_model.evaluate(X_test, y_test)
print(f'Dense Model Test Accuracy: {dense_test_accuracy * 100:.2f}%')



# Define CNN model
cnn_model = models.Sequential([
    layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(64, (3, 3, 3), activation='relu'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(64, (3, 3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # Assuming 2 classes (ADHD and hCon)
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the CNN model on the test set
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f'CNN Model Test Accuracy: {cnn_test_accuracy * 100:.2f}%')
