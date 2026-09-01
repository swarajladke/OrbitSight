Offline inference container for the TII OrbitSight Challenge.

Load:   gunzip -c image.tar.gz | docker load
Image:  orbitsight:latest

Run:
  docker run --rm --network none \
    -v /path/to/OrbitSight_dataset:/OrbitSight_dataset:ro \
    -v /path/to/work:/work \
    orbitsight:latest

Output:      /work/OrbitAI/DDMMYYYY
Mirrors:     /work/orbitai, /work/orbitsight, /work/OrbitSight
Archive:     188,539,903 bytes gzipped, 582,660,096 bytes raw
SHA-256:     7bb765b28dd8e600c4f82969fadfd1449a3b4c57f5cbd76749bb2e323c12c722
Runtime:     CPU-only, requires no network
Built from:  commit bbf68d72b7ec62fd558f52c23399c5d12acfffb6
