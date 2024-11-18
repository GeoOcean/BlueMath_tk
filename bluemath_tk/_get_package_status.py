import os
import re


def check_package_completion(base_dir):
    total_files = 0
    empty_files = 0
    written_files = 0
    files_with_docstrings = 0
    empty_file_list = []
    written_file_list = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                total_files += 1
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) == 0:
                    empty_files += 1
                    empty_file_list.append(file_path)
                else:
                    written_files += 1
                    written_file_list.append(file_path)
                    with open(file_path, "r") as f:
                        content = f.read()
                        if '"""' in content or "'''" in content:
                            files_with_docstrings += 1

    if total_files == 0:
        return 0.0, empty_file_list, written_file_list, files_with_docstrings

    completion_percentage = ((total_files - empty_files) / total_files) * 100
    return (
        completion_percentage,
        empty_file_list,
        written_file_list,
        files_with_docstrings,
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    completion_percentage, empty_file_list, written_file_list, files_with_docstrings = (
        check_package_completion(base_dir)
    )
    print(f"Package completion: {completion_percentage:.2f}%")
    print(f"Empty files ({len(empty_file_list)}):")
    for file in empty_file_list:
        print(f"  {file}")
    print(f"Written files ({len(written_file_list)}):")
    for file in written_file_list:
        print(f"  {file}")
    print(f"Files with docstrings: {files_with_docstrings}")
    readme_path = os.path.join(base_dir, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as readme_file:
            readme_content = readme_file.read()

        new_readme_content = re.sub(
            r"Package completion: \d+\.\d+%",
            f"Package completion: {completion_percentage:.2f}%",
            readme_content,
        )
        with open(readme_path, "w") as readme_file:
            readme_file.write(new_readme_content)
