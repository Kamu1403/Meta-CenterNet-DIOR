#!/usr/bin/env bash

DIR=$1
tensorboard --logdir "$DIR" --bind_all