# macOS Build for AnyLabeling

## Overview

The macOS build of AnyLabeling is provided as a directory structure rather than a bundled `.app` file. This approach offers:

- Easier integration with other tools or scripts
- Customization of the application's resources
- Better compatibility across different macOS versions
- Direct access to application files

## Installation

1. Download the appropriate `AnyLabeling-Folder.zip` (CPU) or `AnyLabeling-Folder-GPU.zip` (GPU) from the [releases page](https://github.com/vietanhdev/anylabeling/releases).

2. Extract the downloaded ZIP file:
   ```bash
   unzip AnyLabeling-Folder.zip
   ```

3. The extracted folder `AnyLabeling-Folder` contains everything needed to run the application.

## Running the Application

To run the application, execute the `anylabeling` binary in the folder:

```bash
cd AnyLabeling-Folder
./anylabeling
```

You can also create a shortcut or alias to this executable for easier access.

## Building Locally

If you want to build the application yourself:

1. Clone the repository:
   ```bash
   git clone https://github.com/vietanhdev/anylabeling.git
   cd anylabeling
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-macos-dev.txt
   ```

3. Make the build script executable:
   ```bash
   chmod +x scripts/build_macos_folder.sh
   ```

4. Run the build script:
   ```bash
   # For CPU version
   ./scripts/build_macos_folder.sh
   
   # For GPU version
   ./scripts/build_macos_folder.sh GPU
   ```

5. The built application folder will be available at `./dist/AnyLabeling-Folder/` (CPU) or `./dist/AnyLabeling-Folder-GPU/` (GPU).

## Troubleshooting

If you encounter issues with the application:

- Ensure you have the correct permissions to execute the application:
  ```bash
  chmod +x AnyLabeling-Folder/anylabeling
  ```

- If you get dynamic library loading errors, make sure all dependencies are properly installed:
  ```bash
  pip install -r requirements-macos.txt
  ```

- If you encounter graphics or UI issues, try using the CPU version instead of GPU.

- For further assistance, please [open an issue](https://github.com/vietanhdev/anylabeling/issues/new) on GitHub. 