#!/bin/sh

wget https://github.com/FortAwesome/Font-Awesome/archive/7.1.0.tar.gz -O font-awesome.tar.gz
tar xf font-awesome.tar.gz
pyrcc5 resources.qrc -o resources.py
rm -i font-awesome.tar.gz
