import tensorflow as tf

# Load model without compiling (ignores optimizer & training config)
model = tf.keras.models.load_model("my_model.h5", compile=False)

# Save in new Keras format (removes batch_shape issues)
model.save("my_model.keras")

print("✅ Model successfully converted to my_model.keras")
