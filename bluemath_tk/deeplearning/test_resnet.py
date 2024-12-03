import keras
from models import resnet_model
from generators.mockDataGenerator import MockDataGenerator

# instantiate model class (load memory)
model = resnet_model.get_model(
              image_height = 64,
              image_width = 64,
              input_frames =  1,
              output_frames =  1)

# print summary of the model
print(model.summary())

# instantiate generator class
train_generator = MockDataGenerator(num_images=5000,
                                       input_height = 64,
                                       input_width = 64,
                                       output_height = 64,
                                       output_width = 64,
                                       batch_size=1)
# define oprimizer
optimizer=keras.optimizers.AdamW
model.compile(
    optimizer=optimizer(
        learning_rate=1e-4, weight_decay=1e-5
    ),
    loss=keras.losses.mean_squared_error,
)

# start the train loop with the fit method
history = model.fit(
    train_generator,
    initial_epoch = 0,
    epochs=20,
    steps_per_epoch=500)


print("training complete")
