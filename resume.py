from tensorflow.keras.models import load_model

model = load_model("stage1_model.h5")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    initial_epoch=4
)