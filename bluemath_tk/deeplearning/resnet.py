import keras
from models import resnet_model
from generators.ncDataGenerator import DataGenerator

# instantiate model class (load memory)
model = resnet_model.get_model(
    image_height=64, image_width=64, input_frames=1, output_frames=1
)

# print summary of the model
print(model.summary())

# instantiate generator class
train_generator = DataGenerator(
    msl_path="/home/tausiaj/DATA/Comparison-ERA5/msl_spain.nc",
    tp_path="/home/tausiaj/DATA/Comparison-ERA5/tp_spain.nc",
    num_images=8760,
    sequential=False,
    batch_size=1,
)

a, b = train_generator.__getitem__(1)
print(a.shape)
print(b.shape)
# define oprimizer
optimizer = keras.optimizers.AdamW
model.compile(
    optimizer=optimizer(learning_rate=1e-4, weight_decay=1e-5),
    loss=keras.losses.mean_squared_error,
)

# start the train loop with the fit method
history = model.fit(train_generator, initial_epoch=0, epochs=20, steps_per_epoch=500)


print("training complete")
