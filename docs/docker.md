# Docker Compose on Linux

Run the desktop application from this checkout without installing Python or
PyQt6 on the host. The image uses Python 3.12 and the project's default CPU
dependencies. NVIDIA hardware, drivers, and the NVIDIA Container Toolkit are
not required.

## Requirements

- A Linux desktop with X11, or a Wayland session with XWayland enabled and a
  working `DISPLAY`. Native Wayland-only sessions are not supported.
- Docker Engine with the Docker Compose v2 plugin and BuildKit, accessible by
  your regular desktop user. Run the commands below without `sudo`.
- Your desktop session's Xauthority file, normally identified by `XAUTHORITY`
  or stored at `~/.Xauthority`.

This setup uses a local X11 Unix socket. Windows, macOS, SSH-forwarded displays,
and Docker daemons on another machine are not supported. Sandboxed Docker
installations, including Docker Desktop, may restrict access to the host's
display socket or authority file. Docker Engine on the Linux host is recommended.
The UID/GID setup assumes rootful Docker Engine without user namespace remapping.
Rootless Docker and `userns-remap` need custom user mappings and are not covered
by this configuration.

## Configure this checkout once

From the repository root, create the default dataset directory and copy the
[example settings](../.env_example) to `.env`. If `.env` already exists, merge
the example settings into it instead of overwriting it.

```bash
mkdir -p data
cp .env_example .env
```

Edit `.env` and set `LOCAL_UID` to the output of `id -u`, `LOCAL_GID` to the
output of `id -g`, and `HOST_HOSTNAME` to the output of `hostname`. The dataset
and authority settings already have defaults for the standard local setup.

Compose reads `.env` automatically on every invocation. It is ignored by Git
and excluded from the Docker build context. The authority path is resolved
from the current desktop session each time, so no temporary cookie file or
repeated exports are needed. Configure each checkout for the user and machine
that will run it. Do not use root's UID/GID.

## Start the application

After the one-time setup, run this command in a terminal inside your graphical
desktop session:

```bash
docker compose up --build
```

The application opens on your desktop. Put images in the repository's `data/`
directory and use **Open Dir** to open `/data` in AnyLabeling.

The host's Xauthority file and X11 socket are mounted read-only. The container
uses the hostname recorded in `.env` so hostname-specific X11 cookies work
without copying or rewriting them. Do not use `xhost +`.

The first build downloads the runtime dependencies. Later builds reuse Docker
layers and a BuildKit pip cache. Dependencies follow `pyproject.toml`, so a
fresh build can resolve newer compatible releases.

## Open an image folder

The folder picker shows the container's filesystem. `/home/anylabeling` is the
application user's home inside the container. Your computer's image folders
become accessible through the shared directory mounted at `/data`.

To use the default shared directory:

1. Copy your images, or a folder containing images, into the repository's `data/`
   directory on your computer.
2. Select **Open Dir** in AnyLabeling.
3. In the folder picker, enter `/data` in the **Directory** field at the bottom
   and click **Choose**. If you copied a subfolder, enter its container path,
   such as `/data/photos`, instead.

| On your computer | In the AnyLabeling folder picker |
| --- | --- |
| Repository directory `data/` | `/data` |
| A subfolder named `data/photos/` | `/data/photos` |

Copying more images into the shared directory does not require rebuilding or
restarting the container. Use **Open Dir** again to load the updated folder.
Annotations saved under `/data` are written to the corresponding directory
on your computer.

## Use another dataset or authority file

To annotate an existing image folder without copying its files, edit
`ANYLABELING_DATA_DIR` in `.env` and set it to that folder's actual absolute path.
Relative paths are resolved from the repository directory. Choose a directory
that your desktop user can read and write. Missing bind-mount sources are
rejected instead of being created as root-owned directories.

After changing the dataset directory, close the application and recreate the
container from the repository root:

```bash
docker compose up --build --force-recreate
```

In AnyLabeling, select **Open Dir** and open `/data` again. The selected host
folder is always mounted at `/data`, even when its host path is different.
Images are read directly from that folder, and annotations saved there remain
available on your computer.

If your desktop does not export `XAUTHORITY` and does not use `~/.Xauthority`,
set `ANYLABELING_XAUTHORITY` in `.env` to the actual session authority file.
Shell variables with the same names override `.env` values. For compatibility,
`DATASET_DIR` and `XAUTHORITY_FILE` are also accepted when their corresponding
`ANYLABELING_` settings are absent.

## Datasets, annotations, and persistent state

| Container path | Storage | Contents |
| --- | --- | --- |
| `/data` | Host directory `ANYLABELING_DATA_DIR`, default `./data` | Images, native JSON annotations, and exports saved here |
| `/home/anylabeling/.anylabelingrc` | Named volume `anylabeling-home` | Application configuration, including custom model references |
| `/home/anylabeling/.config/anylabeling/` | Same named volume | Qt settings, such as window layout and recent paths |
| `/home/anylabeling/anylabeling_data/models/` | Same named volume | Downloaded model weights and their configuration files |
| `/home/anylabeling/.cache/` | Same named volume | User caches, including Hugging Face downloads |

The entire container home is persisted because the application uses several
home-relative paths. Docker initializes a new home volume with the image user's
ownership. The application runs as this non-root user from its first launch.
No startup script changes dataset contents or permissions.

Save annotations in `/data` to keep them directly accessible on the host.
AnyLabeling saves its editable annotations as JSON. Use **Export Annotations**,
select **YOLO (.txt)**, and choose an output directory under `/data` to persist
YOLO detection or segmentation exports on the host as well. Exported images,
labels, and other output files retain the container user's UID/GID.

For custom models, place both the model configuration and weights under `/data`
and select them using their container paths. Host paths outside the mounted
dataset are not visible. Downloads require network access and enough disk space.

The default build UID/GID is `1000:1000`. The local `.env` file records your
actual UID/GID. A home volume belongs to that UID/GID. Reusing it with another UID/GID requires a deliberate ownership
migration or a separate Compose project and volume. For a separate project,
use the same `-p` option for every command, for example
`docker compose -p anylabeling-other up --build`.

## Stop, rebuild, and recreate

Close the application normally to flush its settings. If Compose remains
attached, press `Ctrl+C`. To stop and remove the container:

```bash
docker compose down
```

The dataset and home volume remain available. Rebuild after changing the source:

```bash
docker compose up --build --force-recreate
```

Do not add `--volumes` to `docker compose down` unless you intend to delete the
saved home, including downloaded models and settings. Dataset bind mounts are
independent of that volume.

## Troubleshooting

- **`DISPLAY` is missing or Qt cannot connect to the display:** run the setup
  from a terminal in the local graphical session. On Wayland, confirm XWayland
  is installed and enabled. Check that `/tmp/.X11-unix` contains your display's
  socket. A native Wayland socket alone is insufficient.
- **Authorization required / invalid MIT-MAGIC-COOKIE:** check that the authority
  file exists and belongs to the desktop session. If `xauth` is installed,
  `xauth nlist "$DISPLAY"` can check for an entry without sharing its secret
  output. Some desktops use a file under `/run/user/...`. Check `XAUTHORITY` or
  set `ANYLABELING_XAUTHORITY` explicitly. `HOST_HOSTNAME` must match `hostname`.
  Recreate the container after logging in again or changing the authority file.
- **Qt reports that the `xcb` plugin cannot load:** check the preceding display
  and authentication errors first. For more detail, run
  `docker compose run --rm -e QT_DEBUG_PLUGINS=1 anylabeling`.
- **Missing mount source:** run `mkdir -p data` for the default dataset, or check
  the existing directory selected by `ANYLABELING_DATA_DIR`. Check the authority
  file selected by `ANYLABELING_XAUTHORITY` as well.
- **Permission denied saving annotations:** check that `LOCAL_UID`/`LOCAL_GID`
  in `.env` match `id -u`/`id -g` and that the host user owns or can write the dataset.
  Rebuild after changing these values. Directories accessible only through a
  supplementary host group require an explicit matching `group_add` in a local
  Compose override. Do not use global write permissions as a workaround.
- **SELinux denies access:** the supplied mounts do not relabel host files.
  Use a site-approved policy or a dataset-specific mount override. Do not
  relabel the shared X11 socket or disable SELinux globally.

The container uses the host display with the same access as another local X11
client. It does not use VNC, a browser, host networking, privileged mode, or a
host IPC namespace. Qt's X11 shared-memory extension is disabled so the host IPC
namespace does not need to be shared. CPU inference can be slower than GPU
inference, particularly for large segmentation models.

## Validation before contributing

With the one-time setup complete, run:

```bash
docker compose config
docker compose build
docker compose up
```

Confirm that the actual application window appears. Open an image under `/data`,
draw and save an annotation, then export it as YOLO under `/data`. Verify the
JSON and TXT files on the host and check their ownership with `ls -ln`. Change
a setting and download a model, close the app, then use
`docker compose up --force-recreate`. Confirm the setting and model remain
available and the annotation can be reopened. Run these checks on a CPU-only
machine or without exposing any GPU devices to the container.

An offscreen Qt launch can check imports and runtime libraries, but cannot
establish that the host display works. Report graphical and model-download
checks separately when they cannot be performed.
