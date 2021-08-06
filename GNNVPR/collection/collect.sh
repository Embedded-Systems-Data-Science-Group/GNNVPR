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
ARGS="--route_chan_width 300 -j 6 --collect_data on --gnntype 3 > /dev/null 2>&1"
ARGS_HYPE="--route_chan_width 300 -j 8 --collect_data on"
ARGS_OUT="--route_chan_width 300 -j 6 --output_final_costs on > /dev/null 2>&1"
ARGS_IN="--route_chan_width 300 -j 6 --input_initial_costs on > /dev/null 2>&1"
ARGSN="--route_chan_width 300 -j 6 > /dev/null 2>&1"
ARGSR="--route_chan_width 300 -j 6 --collect_data on > /dev/null 2>&1"

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

# 1. Name, 2. Architecture Directory, 3. Benchmarks, 4. Architecture, 5. VTR Arguments
# run_benchmark "EARCH MCNC" "$arch_dir2" "$MCNC" "$EARCH" "$ARGSR"
run_benchmark "EARCH MCNC" "$arch_dir2" "$MCNC" "$EARCH" "$ARGSR"
run_benchmark "STRATXIV MCNC" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGSR"
run_benchmark "STRATXIV TITAN" "$arch_dir1" "$TITAN" "$STRATXIV" "$ARGSR"
run_benchmark "STRATXIV TITANJR" "$arch_dir1" "$TITANJR" "$STRATXIV" "$ARGSR"


# run_benchmark "STRATXIV MCNC" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGSR"
# run_benchmark "STRATXIV TITANJR" "$arch_dir1" "$TITANJR" "$STRATXIV" "$ARGSR"

