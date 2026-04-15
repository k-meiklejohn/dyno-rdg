# Dyno-RDG Generator

## Installation

### Precompiled Binaries

Standalone executable files are available for MacOS, Windows and Linux (compiled on Ubuntu 22.04) These can be downloaded from the releases section of the github [here](https://github.com/k-meiklejohn/dyno-rdg/releases). 

#### Linux (Ubuntu):

Once the executable is downloaded, it must be made executable. This can be done in Ubuntu by right clicking, choosing Properties, and toggling 'Executable as Program' to on.
Alternatively with the command line:

```bash
chmod +x dyno-rdg-linux
```

The app can then be run by double clicking the file, or running

```bash
./dyno-rdg-linux
```

Running from terminal also allows you to see the edgelist from which the graph is derived


#### MacOS

The has a quarantine attribute when downloaded. To overcome this control-click the executable and choose open. Choose open in the following dialogue box.
To permanently remove the quarantine attribute open the terminal and run:

```bash
sudo xattr -r -d com.apple.quarantine dyno-rdg-mac
```

#### Windows
