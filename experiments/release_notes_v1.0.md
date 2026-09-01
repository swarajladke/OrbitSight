Offline inference container for the TII OrbitSight Challenge.

Load:   gunzip -c image.tar.gz | docker load
Image:  orbitsight:latest

Run:
  docker run --rm --network none \
    -v /path/to/OrbitSight_dataset:/OrbitSight_dataset:ro \
    -v /path/to/work:/work \
    orbitsight:latest

Output:      /work/orbitsight/DDMMYYYY
Archive:     186,993,701 bytes gzipped, 574,480,384 bytes raw
SHA-256:     ee469c9c9ecda9f5b760b6e24b39ed201c26af8fd32d3755906a08bc4a28a31c
Runtime:     CPU-only, requires no network
Built from:  commit bbf68d72b7ec62fd558f52c23399c5d12acfffb6
