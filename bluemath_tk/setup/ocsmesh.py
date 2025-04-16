import subprocess
import sys


def post_install():
    """
    Post-installation script to set up OCSMesh dependencies.
    This installs jigsawpy via conda and ocsmesh via pip.
    """
    print("Setting up OCSMesh and dependencies...")

    try:
        # Install jigsawpy from conda-forge
        print("Installing jigsawpy from conda-forge...")
        subprocess.check_call(
            ["conda", "install", "-y", "-c", "conda-forge", "jigsawpy"]
        )

        # Install ocsmesh via pip
        print("Installing ocsmesh via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ocsmesh"])

        print("Successfully set up OCSMesh and dependencies!")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}", file=sys.stderr)
        print("Command output:", e.output, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error setting up OCSMesh: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(post_install())
