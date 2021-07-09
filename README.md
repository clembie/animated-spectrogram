# animated-spectrogram (fork)
Converts your audio to an animated spectrogram.
An example can be found in my (original author's) [YouTube video](https://www.youtube.com/watch?v=H4UnyiCxFfE).

## Requirements
* Currently, this works only with **mono** wave files.
* Python 2 -- added a Pipfile for dependency installation with `pipenv`
* configparser, scipy, opencv, numpy
* ffmpeg (this one you'll need to install globally on your host machine)

## Usage
* The important parameters are defined in options.cfg.
* Run `python main.py` (automatically loads options.cfg)

## New
* added a function to do linear (as opposed to logarithmic) frequency bands for the FFT. This is potentially more appropriate when doing a video for non-musical material.

## To do
* Parameterize the linear vs logarithmic function
