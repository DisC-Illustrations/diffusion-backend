# diffusion-backend

## Docker

When using Docker, you can build the image with the following command:

```bash
docker build -t diffusion-backend .
```

Then, you can run the image with the following command:

```bash
docker run -p 5000:<local_port> diffusion-backend
```

Replace `<local_port>` with the port you want to use locally. E.g.:

```bash
docker run -p 5000:55000 diffusion-backend
```

## Pytorch and CUDA

If you want to use Pytorch with CUDA on Windows, you have to install the CUDA version of Pytorch.

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
