# convert_model.py (Attempt to fix pickling error)

import tensorflow as tf
import os
import json
from tensorflow.keras import layers
# Import the specific preprocess_input function to be used in the custom layer
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_specific_preprocess_input

# Custom Layer Definition
# This definition needs to be available when model_from_json is called.
# The @register_keras_serializable decorator helps Keras find this class.
@tf.keras.utils.register_keras_serializable()
class PreprocessInputLayer(layers.Layer):
    # When Keras reconstructs this layer from config.json using model_from_json,
    # it will pass the 'name' (e.g., "preprocess_input") and other standard layer
    # arguments (like 'trainable', 'dtype') from the config into **kwargs.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # It's crucial not to store non-serializable objects (like modules or complex functions)
        # as instance attributes (e.g., self.some_module = module) here.

    def call(self, inputs):
        # Use the specifically imported preprocessing function.
        # This function is not stored as part of the layer's state.
        return resnet50_specific_preprocess_input(inputs)

    def get_config(self):
        # Return the base configuration.
        # All necessary parameters (like 'name', 'trainable', 'dtype') are
        # handled by super() based on the kwargs passed during reconstruction
        # from the loaded config.json. This layer adds no new *configurable* parameters.
        base_config = super().get_config()
        return base_config

# --- Main script execution ---
print("--- Starting Manual Model Conversion ---")

# Define paths to the model's components
# Ensure these paths are correct relative to where you run convert_model.py
model_dir_relative_path = os.path.join('models', 'best_model.keras')
config_file_relative_path = os.path.join(model_dir_relative_path, 'config.json')
weights_file_relative_path = os.path.join(model_dir_relative_path, 'model.weights.h5')

# Define the final path for the single .h5 file
h5_output_file_relative_path = os.path.join('models', 'best_model.h5')

# Get absolute paths for clarity in print statements and error messages
base_dir = os.getcwd()
model_dir_abs = os.path.join(base_dir, model_dir_relative_path)
config_path_abs = os.path.join(base_dir, config_file_relative_path)
weights_path_abs = os.path.join(base_dir, weights_file_relative_path)
h5_file_path_abs = os.path.join(base_dir, h5_output_file_relative_path)


try:
    print(f"1. Reading model architecture from: {config_path_abs}")
    if not os.path.exists(config_path_abs):
        raise FileNotFoundError(f"Config file not found: {config_path_abs}")
    with open(config_path_abs, 'r') as json_file:
        model_json_string = json_file.read()
    
    # Recreate the model from the JSON string.
    # Keras needs to know about 'PreprocessInputLayer'. The decorator and/or custom_objects help.
    custom_objects_map = {'PreprocessInputLayer': PreprocessInputLayer}
    model = tf.keras.models.model_from_json(model_json_string, custom_objects=custom_objects_map)
    print("   - Architecture loaded successfully.")

    print(f"2. Loading model weights from: {weights_path_abs}")
    if not os.path.exists(weights_path_abs):
        raise FileNotFoundError(f"Weights file not found: {weights_path_abs}")
    model.load_weights(weights_path_abs)
    print("   - Weights loaded successfully.")

    print(f"3. Saving the complete model to: {h5_file_path_abs}")
    # include_optimizer=False is good practice if only doing inference with the saved model.
    model.save(h5_file_path_abs, include_optimizer=False) 

    print("\n--- Conversion Complete! ---")
    print(f"A new file '{os.path.basename(h5_file_path_abs)}' has been created in the '{os.path.dirname(h5_file_path_abs)}' folder.")
    print("Please ensure your app/main.py now points to this '.h5' file.")

except FileNotFoundError as fnf_error:
    print(f"\nFile Not Found Error: {fnf_error}")
    print("Please double-check the paths to your model files.")
except Exception as e:
    print(f"\nAn error occurred during model conversion: {e}")
    print("Details of the error are above. Please check TensorFlow/Keras versions and compatibility.")

