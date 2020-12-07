#!/bin/bash

# Use this for generating standard results for hardware

exe=./bench

echo " "
echo "Size     1 thr  2 thr  4 thr"
echo "---------------------------------"
echo "SMALL:  "
$exe -silent -epochs 10 -pats 100 -units 25 $*
$exe -silent -epochs 10 -pats 100 -units 25 -threads=2 $*
$exe -silent -epochs 10 -pats 100 -units 25 -threads=4 $*
echo "MEDIUM: "
$exe -silent -epochs 3 -pats 100 -units 100 $*
$exe -silent -epochs 3 -pats 100 -units 100 -threads=2 $*
$exe -silent -epochs 3 -pats 100 -units 100 -threads=4 $*
echo "LARGE:  "
$exe -silent -epochs 5 -pats 20 -units 625 $*
$exe -silent -epochs 5 -pats 20 -units 625 -threads=2 $*
$exe -silent -epochs 5 -pats 20 -units 625 -threads=4 $*
echo "HUGE:   "
$exe -silent -epochs 5 -pats 10 -units 1024 $*
$exe -silent -epochs 5 -pats 10 -units 1024 -threads=2 $*
$exe -silent -epochs 5 -pats 10 -units 1024 -threads=4 $*
echo "GINORM: "
$exe -silent -epochs 2 -pats 10 -units 2048 $*
$exe -silent -epochs 2 -pats 10 -units 2048 -threads=2 $*
$exe -silent -epochs 2 -pats 10 -units 2048 -threads=4 $*

