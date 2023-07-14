#!/bin/bash

# Download LJSPeech dataset LJSpeech-1.1.tar.bz2
tar xvf LJSpeech-1.1.tar.bz2
rm -rf LJSpeech-1.1.tar.bz2
ln -s LJSpeech-1.1/wavs DUMMY1

# Download background_noise.tar
tar xvf background_noise.tar
rm -f background_noise.tar
