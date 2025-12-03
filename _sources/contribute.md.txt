# Contributions

Thank you for considering contributing to the BlueMath_tk package! We welcome contributions from the community and are grateful for your support.

## How to Contribute

1. Fork the Repository (skip if you are GeoOcean).

Start by forking the repository to your GitHub account. This will create a copy of the repository under your GitHub account.

2. Clone the Repository.

Clone the forked repository to your local machine using the following command:
```sh
git clone https://github.com/YOUR-USERNAME/BlueMath_tk.git
```

3. Create a branch. 

Create a new branch for your contribution. Use a descriptive name for your branch:
```sh
git checkout -b feature/your-feature-name
```
An example branch could be `feature/xbeach-wrapper`.

4. Make changes.

Make the necessary changes to the codebase. Ensure that your code follows the project's coding standards and guidelines.

5. Commit changes.

```sh
git add .
git commit -m "Add your commit message here"
```

6. Push changes.

Push your changes to your forked repository (Remember repo is not forked if your GeoOcean):
```sh
git push origin feature/your-feature-name
```

7. Create a Pull Request.

Go to the original repository on GitHub and create a pull request from your forked repository. Provide a clear and detailed description of your changes and the problem they solve.

8. Review Process.

Your pull request will be reviewed by the maintainers. Please be responsive to any feedback or questions they may have. Once your pull request is approved, it will be merged into the main branch.

9. Reporting Issues.

If you encounter any issues or bugs, please report them by creating a new issue on the GitHub repository. Provide as much detail as possible to help us understand and resolve the issue.

10. Code of Conduct.

Please adhere to our Code of Conduct to ensure a welcoming and inclusive environment for everyone.

11. License.

By contributing to the BlueMath_tk package, you agree that your contributions will be licensed under the GPL-3.0 License.

12. Contact

If you have any questions or need further assistance, feel free to reach out to the maintainers.

Thank you for your contributions and support!

## Documentation

When creating new `python` code, it is essential to properly document all new classes and functions. Below, we show how the **docstrings** of classes should look, so the community can properly learn how to use **BlueMath**.

Code example:
```python
import pickle

from .models import BlueMathModel


def load_model(model_path: str) -> BlueMathModel:
    """
    Loads a BlueMathModel from a file.

    Parameters
    ----------
    model_path : str
        The path to the model file.

    Returns
    -------
    BlueMathModel
        The loaded BlueMathModel.
    """

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model
```