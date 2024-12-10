# Model Wrappers

This section provides general documentation for the model wrappers usage. The wrappers are designed to facilitate the interaction with various numerical models by providing a consistent interface for setting parameters, running simulations, and processing outputs.

For more detailed information, refer to the specific class implementations and their docstrings.

## BaseModelWrapper

The `BaseModelWrapper` class serves as the base class for all model wrappers. It provides common functionality that can be extended by specific model wrappers.

## SwashModelWrapper

The `SwashModelWrapper` class is a specific implementation of the `BaseModelWrapper` for the SWASH model. It extends the base functionality to handle SWASH-specific requirements.

### Example Usage

```python
import numpy as np
from bluemath_tk.wrappers.swash_wrapper import SwashModelWrapper

# Initialize the SWASH model wrapper
template_dir = "path/to/templates"
input_template = "swash_input_template.j2"
output_dir = "path/to/output"
parameters = {
    "vegetation_height": 1.5,
    "param2": "value2",
    "param3": "value3",
}

swash_model = SwashModelWrapper(template_dir, input_template, output_dir, **parameters)

# Set the SWASH executable path
swash_model.set_swash_exec("/path/to/swash/executable")

# Render the input file
context = {"param1": "value1", "param2": "value2"}
swash_model.render_input_file(context, "input_file.sws")

# Run the model
swash_model.run_model("input_file.sws")

# Write an array to a file
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
swash_model.write_array_in_file(array_2d, "array_2d.txt")
```
