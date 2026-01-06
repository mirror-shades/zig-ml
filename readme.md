# Zig ML project

This is an OCR ML library rewritten in Zig. It was written to be more ideomatic to the Zig language, using error handling and struct methods. It reduces LOC by a couple hundred lines.

Original project:  
https://www.youtube.com/watch?v=hL_n_GljC0I  
https://github.com/Magicalbat/videos/tree/main/machine-learning

## Usage

### Preparing the MNIST Dataset

First, download and convert the MNIST dataset to binary format. You will need a few libs so using venv is recommended:

```bash
python mnist.py
```

This will generate the following binary files in the project root:
- `train_images.bin`
- `train_labels.bin`
- `test_images.bin`
- `test_labels.bin`

**Note:** Requires `tensorflow-datasets` and `numpy`:
```bash
pip install tensorflow-datasets numpy
```

### Compiling

Build and run the project using Zig 0.15.2 compiler:

```bash
zig build run
```

