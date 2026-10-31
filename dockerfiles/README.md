# CI container images

Each `<tag>.Dockerfile` here builds one image, `ghcr.io/accel-sim/accel-sim-framework:<tag>`.
These files used to live one per branch in `accel-sim/Dockerfile`, which is now retired.

The `*-minimal` images are what CI runs (`IMAGE_TAG` in `.github/workflows/main.yml`). They hold:
- the build dependencies
- the gpu-app-collection CI apps and their regression data
- the AWS CLI and boto3, so the image can also run AWS Batch jobs (`util/aws_batch`) — 12.8-minimal only so far

## When images build

`.github/workflows/docker-images.yml` builds only the Dockerfiles a change touches, and lists each one as build or skip with a reason. The rules are in `.github/scripts/docker-plan.sh`:
- **Pull request:** builds the images whose Dockerfile changed, without pushing.
  - A modified file always builds.
  - A new file builds only if its tag is not on ghcr yet.
- **Merge to `dev`:** the same rules, followed by a push to ghcr.
  - The build runs on the same docker daemon as the PR's build, so it is a cache hit and the push is the only real work.
- **Manual:** Actions → Docker images → Run workflow, with the tags to build. Tick `no_cache` to rebuild from scratch, for example to pick up a newer gpu-app-collection.

## Build host

Builds run on a `tgrogers-raid` runner, which has no docker daemon. Its account (`tgrghci`) has the docker CLI and buildx installed. The workflow sets `DOCKER_HOST=ssh://tgrogers-pc02`, so the build runs on tgrogers-pc02's daemon.

To build by hand on any docker host:

```bash
docker build -f dockerfiles/Ubuntu-24.04-cuda-12.8-minimal.Dockerfile dockerfiles
```
