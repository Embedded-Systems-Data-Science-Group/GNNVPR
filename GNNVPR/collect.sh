#!/bin/bash
# Output Folder

# VPR Command
BASEDIRECTORY="/mnt/e/benchmarks"
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
ARGSGB="-j 16 --verify_binary_search on --gnntype 1"
ARGSDB="-j 16 --verify_binary_search on --gnntype 0"

# Arguments
ARGSD="-j 16 --gnntype 0 --route_chan_width"
ARGSR="-j 16 --collect_data on --route_chan_width" 
ARGSG="-j 16 --gnntype 1 --route_chan_width"
ARGSH="-j 16 --gnntype 2 --route_chan_width"
ARGSO="-j 16 --output_final_costs on --route_chan_width"
ARGSI="-j 16 --gnntype 3 --route_chan_width"

# Ignore Output
ERR="> /dev/null 2>&1"
# ERR=""
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
            rm -f "$_mydir"/inference/processed/*
            rm -f "$_mydir"/inference/raw/*.csv
            # eval python script with circuit & arch path -> get values back.
            echo "$(pwd)"
            MINW=$(python /home/spicygremlin/Github/CS220/GNNVPR/minw.py -a $4 -c $b 2>&1)
            eval "$INITIAL $4 $b $5 ${MINW} $ERR"
            cd "$OUTPUT"
        done
    done

}

run_benchmark2() {
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
            rm -f "$_mydir"/inference/processed/*
            rm -f "$_mydir"/inference/raw/*.csv
            # eval python script with circuit & arch path -> get values back.
            echo "$(pwd)"
            # MINW=$(python /home/spicygremlin/Github/CS220/GNNVPR/minw.py -a $4 -c $b 2>&1)
            # eval "$INITIAL $4 $b $5 ${MINW} $ERR"
            eval "$INITIAL $4 $b $5 $ERR"
            cd "$OUTPUT"
        done
    done

}
# Creates CSV with header, all benchmarks append to this. 
# collect_minw_header() {

#     echo "architecture,circuit,minw" > "/benchmarks/minw_data/minw.csv"

# }
# collect_minw_header
# run_benchmark "EARCH MCNC MINW" "$arch_dir2" "$MCNC" "$EARCH" "$ARGSR"
# run_benchmark "STRATXIV MCNC MINW" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGSR"


# run_benchmark "STRATXIV TITANJR MINW" "$arch_dir1" "$TITANJR" "$STRATXIV" "$ARGSR"
# run_benchmark "STRATXIV TITAN MINW" "$arch_dir1" "$TITAN" "$STRATXIV" "$ARGSR"

# 1. Name, 2. Architecture Directory, 3. Benchmarks, 4. Architecture, 5. VTR Arguments
# run_benchmark "EARCH MCNC MINW" "$arch_dir2" "$MCNC" "$EARCH" "$ARGSO"
# run_benchmark "STRATXIV MCNC MINW" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGSO"

# run_benchmark "STRATXIV MCNC MINW" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGSG"


# TODO: This one
# run_benchmark "EARCH MCNC GNN Binary Search" "$arch_dir2" "$MCNC" "$EARCH" "$ARGSD"
run_benchmark "EARCH MCNC GNN Default" "$arch_dir2" "$MCNC" "$EARCH" "$ARGSG"


# run_benchmark "STRATXIV MCNC Default" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGSD"
# run_benchmark "STRATXIV MCNC GNN" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGSG"

# run_benchmark2 "EARCH MCNC GNN Binary Search" "$arch_dir2" "$MCNC" "$EARCH" "$ARGSDB"
# run_benchmark2 "EARCH MCNC GNN Binary Search" "$arch_dir2" "$MCNC" "$EARCH" "$ARGSGB"



# run_benchmark "EARCH MCNC" "$arch_dir2" "$MCNC" "$EARCH" "$ARGS"
# run_benchmark "EARCH MCNC" "$arch_dir2" "$MCNC" "$EARCH" "$ARGS_HYPE"
# run_benchmark "STRATXIV MCNC" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGS_HYPE"
# run_benchmark "STRATXIV TITAN" "$arch_dir1" "$TITAN" "$STRATXIV" "$ARGS_HYPE"
# run_benchmark "STRATXIV TITANJR" "$arch_dir1" "$TITANJR" "$STRATXIV" "$ARGS_HYPE"


# run_benchmark "STRATXIV MCNC" "$arch_dir1" "$MCNC" "$STRATXIV" "$ARGSR"
# run_benchmark "STRATXIV TITANJR" "$arch_dir1" "$TITANJR" "$STRATXIV" "$ARGSR"

