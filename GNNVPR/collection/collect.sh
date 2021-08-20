#!/bin/bash
# Output Folder

# VPR Command
BASEDIRECTORY="/benchmarks"
OUTPUT="$BASEDIRECTORY/Outputs/"
INITIAL="$VTR_ROOT/vpr/vpr"
# Architectures
EARCH="$BASEDIRECTORY/arch/MCNC/EArch.xml"
STRATXIV="$BASEDIRECTORY/arch/TITAN/stratixiv_arch.timing.xml"
# Benchmarks
TITAN="$BASEDIRECTORY/blif/TITAN/*.blif"
MCNC="$BASEDIRECTORY/blif/MCNC/*.blif"
TITANJR="$BASEDIRECTORY/blif/TITANJR/*.blif"

# Argument Options
# TODO: Refactor to be more modular. 
ARGS_MINW="-j 6 --collect_data off --collect_metrics off --gvminw on > /dev/null 2>&1"
ARGS="-j 6 --gnntype 1 > /dev/null 2>&1 "
ARGS_HYPE=" -j 8 --gnntype 2 > /dev/null 2>&1 "
ARGS_OUT="--route_chan_width 300 -j 6 --output_final_costs on > /dev/null 2>&1"
ARGS_IN="--route_chan_width 300 -j 6 --input_initial_costs on > /dev/null 2>&1"
ARGSN="--route_chan_width 300 -j 6 > /dev/null 2>&1"
ARGSR="-j 16 --collect_data on > /dev/null 2>&1"

cd "$OUTPUT"
IFS="/." read -a arch <<< "$STRATXIV"
arch_dir1=${arch[-3]}
IFS="/." read -a arch <<< "$EARCH"
arch_dir2=${arch[-2]}

# Arguments: Name, Architecture(xml file), Benchmarks(blif files).
run_benchmark() {
    echo "Running $1"
    for arch_dir in "$2";
    do
        for b in $3;
        do  
            IFS="/." read -a bench <<< "$b"
            curdir=$arch_dir"_"${bench[-2]}
            mkdir -p $curdir
            mkdir -p $curdir/inference/
            mkdir -p $curdir/inference/processed/
            mkdir -p $curdir/inference/raw/
            mkdir -p $curdir/inference/combined/
            cd $curdir
            _mydir="$(pwd)"
            rm -f "$_mydir"/inference/*.csv
            rm -f "$_mydir"/inference/raw/*.csv
            eval "$INITIAL $4 $b $5"
            cd "$OUTPUT"
        done
    done

}
# Creates CSV with header, all benchmarks append to this. 
collect_minw_header() {

    echo "architecture,circuit,minw" > "/benchmarks/minw_data/minw.csv"

}
collect_minw_header
run_benchmark "EARCH MCNC MINW" "$arch_dir2" "$MCNC" "$EARCH" "$ARGS_MINW"
run_benchmark "STRATXIV MCNC MINW" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGS_MINW"
run_benchmark "STRATXIV TITANJR MINW" "$arch_dir1" "$TITANJR" "$STRATXIV" "$ARGS_MINW"
run_benchmark "STRATXIV TITAN MINW" "$arch_dir1" "$TITAN" "$STRATXIV" "$ARGS_MINW"

# 1. Name, 2. Architecture Directory, 3. Benchmarks, 4. Architecture, 5. VTR Arguments

# run_benchmark "EARCH MCNC" "$arch_dir2" "$MCNC" "$EARCH" "$ARGS"
# run_benchmark "EARCH MCNC" "$arch_dir2" "$MCNC" "$EARCH" "$ARGS_HYPE"
# run_benchmark "STRATXIV MCNC" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGS_HYPE"
# run_benchmark "STRATXIV TITAN" "$arch_dir1" "$TITAN" "$STRATXIV" "$ARGS_HYPE"
# run_benchmark "STRATXIV TITANJR" "$arch_dir1" "$TITANJR" "$STRATXIV" "$ARGS_HYPE"


# run_benchmark "STRATXIV MCNC" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGSR"
# run_benchmark "STRATXIV TITANJR" "$arch_dir1" "$TITANJR" "$STRATXIV" "$ARGSR"

