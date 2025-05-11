import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r"C:\Users\valli\Downloads\sleepdata.csv")
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True).astype(int)
data['Systolic'] = pd.to_numeric(data['Systolic'])
data['Diastolic'] = pd.to_numeric(data['Diastolic'])
columns_to_normalize = [
    'Age',
    'Sleep Duration',
    'Quality of Sleep',
    'Physical Activity Level',
    'Stress Level',
    'Heart Rate',
    'Daily Steps',
    'Sleep Onset Latency',
    'Systolic',
    'Diastolic',
]
# Initialize the scaler
scaler = MinMaxScaler()
# Normalize the specified columns
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
data.to_csv('normalized_data.csv', index=False)
features = ['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category','Heart Rate','Systolic','Diastolic','Sleep Onset Latency','Iron level']
target = 'Sleep Disorder'  # Assuming this is your target column
# Create a new DataFrame with only the selected features and the target
selected_data = data[features + [target]]
# Check the selected data
print(selected_data.head())
# Encode categorical variables
label_encoder_gender = LabelEncoder()
label_encoder_bmi = LabelEncoder()
label_encoder_iron = LabelEncoder()
label_encoder_disorder = LabelEncoder()
selected_data.loc[:, 'Gender'] = label_encoder_bmi.fit_transform(selected_data['Gender'])
selected_data.loc[:, 'BMI Category'] = label_encoder_bmi.fit_transform(selected_data['BMI Category'])
selected_data.loc[:, 'Iron level'] = label_encoder_bmi.fit_transform(selected_data['Iron level'])

selected_data.loc[:, 'Sleep Disorder'] = label_encoder_disorder.fit_transform(selected_data['Sleep Disorder'])
#classes = label_encoder_disorder.classes_ 
# Get classes as a list of strings
classes = label_encoder_disorder.classes_.astype(str).tolist()  # Convert to list of strings
# Check the selected data after encoding
print(selected_data.head())
# Save the selected features to a new CSV file if needed
selected_data.to_csv('selected_features_encoded.csv', index=False)
stratify_target = selected_data['Iron level']
# Split the dataset into features and target
X = selected_data.drop(columns=['Sleep Disorder'])  # Features
y = selected_data['Sleep Disorder']  # Target
# Ensure that y is of integer type
y = y.astype(int)
# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=stratify_target)
# Convert y_train to a NumPy array of integers
y_train = y_train.values.astype(int)
# Check the shapes of the resulting datasets
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
# After the train-test split
print("Number of rows in training set:", X_train.shape[0])
print("Number of rows in testing set:", X_test.shape[0])
# Check unique classes
print("Unique classes in y_train:", np.unique(y_train))
print("Unique values in 'Sleep Disorder':", data['Sleep Disorder'].unique())
# Print the mapping of sleep disorders to their encoded values
disorder_mapping = {i: disorder for i, disorder in enumerate(label_encoder_disorder.classes_)}
print("Mapping of sleep disorders to encoded values:", disorder_mapping)
# Build the ANN model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Input layer
    layers.Dense(64, activation='relu'),       # Hidden layer 1
    layers.Dense(32, activation='relu'),       # Hidden layer 2
    layers.Dense(len(label_encoder_disorder.classes_), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
#save the model
model.save('model.keras')  # Use the .keras extension
# Get predictions for the test set
predictions = model.predict(X_test)
# Convert predictions to class labels
predicted_classes = predictions.argmax(axis=1)
# Ensure that y_test is transformed back to disorder names
true_labels = label_encoder_disorder.inverse_transform(y_test)
# Map predicted classes back to the original sleep disorder names
predicted_disorders = label_encoder_disorder.inverse_transform(predicted_classes)
# Create a DataFrame to show the results
results = pd.DataFrame({
    'Person ID': X_test.index,  # Assuming the original DataFrame index represents Person IDs
    'True Label': true_labels,   # True disorder names
    'Predicted Disorder': predicted_disorders  # Predicted disorder names
})
# Display all rows
pd.set_option('display.max_rows', None)  # Show all rows in the output
print(results)
# Check if y_test is a NumPy array and convert to int if necessary
if not isinstance(y_test, np.ndarray):
    y_test = np.array(y_test).astype(int)
# Check types
print("X_test dtype:", X_test.dtypes)
print("y_test dtype:", y_test.dtype)
# Create evaluation metrics
print("\nClassification Report:")
try:
    print(classification_report(y_test, predicted_classes, target_names=classes))
except Exception as e:
    print("Error during classification report:", e)
# Confusion matrix
confusion_mtx = confusion_matrix(y_test, predicted_classes)
# Plotting confusion matrix
plt.figure(figsize=(9, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes,  # Actual class names on the x-axis
            yticklabels=classes)  # Actual class names on the y-axis
plt.ylabel('Actual Sleep Disorder')  # Label for y-axis
plt.xlabel('Predicted Sleep Disorder')  # Label for x-axis
plt.title('Confusion Matrix')
plt.show()
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
# Count the occurrences of each predicted disorder
predicted_counts = pd.Series(predicted_disorders).value_counts()
# Create a DataFrame for plotting
predicted_df = predicted_counts.reset_index()
predicted_df.columns = ['Predicted Disorder', 'Count']  # Rename columns for clarity
# Bar plot for predicted sleep disorders
plt.figure(figsize=(9, 6))
sns.barplot(x='Predicted Disorder', y='Count', data=predicted_df, color='skyblue')
plt.title('Distribution of Predicted Sleep Disorders')
plt.xlabel('Predicted Disorder')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Count occurrences of each encoded sleep disorder
disorder_counts = selected_data['Sleep Disorder'].value_counts()
# Calculate total count
total_count = len(selected_data)
# Calculate percentages
disorder_percentages = (disorder_counts / total_count) * 100
# Create a DataFrame for better visualization
disorder_summary = pd.DataFrame({
    'Encoded': disorder_counts.index,  # These are the encoded values (integers)
    'Count': disorder_counts.values,
    'Percentage': disorder_percentages.values
})

# Map the encoded values back to sleep disorder names using the LabelEncoder
disorder_summary['Sleep Disorder'] = label_encoder_disorder.inverse_transform(disorder_summary['Encoded'].astype(int))
# Rearrange the columns for clarity
disorder_summary = disorder_summary[['Sleep Disorder', 'Count', 'Percentage']]
# Display the summary
print(disorder_summary)
# Load your existing model (if saved)
model = keras.models.load_model('model.keras')# Make sure to save your model a
# Load and preprocess training data for normalization and encoding
training_data = pd.read_csv(r"C:\Users\valli\Downloads\sleepdata.csv")
training_data[['Systolic', 'Diastolic']] = training_data['Blood Pressure'].str.split('/', expand=True).astype(int)

# Normalize the same columns used for training
columns_to_normalize = [
    'Age', 'Sleep Duration', 'Quality of Sleep', 
    'Physical Activity Level', 'Stress Level', 
    'Heart Rate', 'Daily Steps', 'Sleep Onset Latency', 
    'Systolic', 'Diastolic'
]
scaler = MinMaxScaler()
training_data[columns_to_normalize] = scaler.fit_transform(training_data[columns_to_normalize])
# Fit label encoders on your training data
label_encoder_gender = LabelEncoder()
label_encoder_bmi = LabelEncoder()
label_encoder_iron = LabelEncoder()
label_encoder_disorder = LabelEncoder()
label_encoder_gender.fit(training_data['Gender'])
label_encoder_bmi.fit(training_data['BMI Category'])
label_encoder_iron.fit(training_data['Iron level'])
label_encoder_disorder.fit(training_data['Sleep Disorder'])
# Collect user input
feature_names = [
    'Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration', 
    'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 
    'BMI Category', 'Blood Pressure', 'Heart Rate', 
    'Daily Steps', 'Sleep Onset Latency', 'Iron level'
]
user_input = []
print("Please enter the following information:")
for feature in feature_names:
    value = input(f"{feature}: ")
    user_input.append(value)
# Convert user input to DataFrame
input_df = pd.DataFrame([user_input], columns=feature_names)
input_df[['Systolic', 'Diastolic']] = input_df['Blood Pressure'].str.split('/', expand=True).astype(int)
# Normalize user input
input_df[columns_to_normalize] = scaler.transform(input_df[columns_to_normalize])
# Prepare the DataFrame with selected features for prediction
features = [
    'Age', 'Gender', 'Sleep Duration', 'Quality of Sleep', 
    'Physical Activity Level', 'Stress Level', 'BMI Category',
    'Heart Rate', 'Systolic', 'Diastolic', 
    'Sleep Onset Latency', 'Iron level'
]
selected_data = input_df[features].copy()
# Encode categorical variables
selected_data['Gender'] = label_encoder_gender.transform(selected_data['Gender'])
selected_data['BMI Category'] = label_encoder_bmi.transform(selected_data['BMI Category'])
selected_data['Iron level'] = label_encoder_iron.transform(selected_data['Iron level'])
# Make predictions using the model
predicted_probabilities = model.predict(selected_data)
# Ensure the model output is as expected
predicted_class = np.argmax(predicted_probabilities, axis=1)
# Map the predicted class back to the original disorder name
predicted_disorder = label_encoder_disorder.inverse_transform(predicted_class)
print(f"The predicted sleep disorder is: {predicted_disorder[0]}")
