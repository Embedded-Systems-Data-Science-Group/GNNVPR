#!/bin/bash
OUTPUT="/mnt/e/benchmarks/Outputs/"
EARCH="/mnt/e/benchmarks/arch/MCNC/EArch.xml"
STRATXIV="/mnt/e/benchmarks/arch/TITAN/stratixiv_arch.timing.xml"
TITAN="/mnt/e/benchmarks/blif/TITAN/*.blif"
MCNC="/mnt/e/benchmarks/blif/MCNC/*.blif"
TITANJR="/mnt/e/benchmarks/blif/TITANJR/*.blif"
INITIAL="$VTR_ROOT/vpr/vpr"
ARGS="--route_chan_width 300 -j 6 --collect_data on --do_inference on > /dev/null 2>&1"
ARGS_OUT="--route_chan_width 300 -j 6 --outtake_ground_truth on > /dev/null 2>&1"
ARGS_IN="--route_chan_width 300 -j 6 --intake_ground_truth on > /dev/null 2>&1"

ARGSR="--route_chan_width 300 -j 6 --collect_data on > /dev/null 2>&1"

cd /mnt/e/benchmarks/Outputs/
# cd $TITAN
IFS="/." read -a arch <<< "$STRATXIV"
arch_dir1=${arch[-3]}
IFS="/." read -a arch <<< "$EARCH"
arch_dir2=${arch[-2]}
# echo "Executing Outtake Ground Truth"
# # # ################################################## OUTTAKE GROUND TRUTH ##############################
# echo "Running EARCH MCNC"
# for arch_dir in "$arch_dir2";
# do
#     for b in $MCNC;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $EARCH $b $ARGS_OUT "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done


# echo "Running STRATXIV MCNC"
# for arch_dir in "$arch_dir1";
# do
#     for b in $MCNC;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $STRATXIV $b $ARGS_OUT "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done
# echo "Executing Intake of Ground Truth"

# ################################################## INTAKE GROUND TRUTH ##############################
# echo "Running EARCH MCNC"
# for arch_dir in "$arch_dir2";
# do
#     for b in $MCNC;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $EARCH $b $ARGS_IN "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done


# echo "Running STRATXIV MCNC"
# for arch_dir in "$arch_dir1";
# do
#     for b in $MCNC;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $STRATXIV $b $ARGS_IN "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done

# echo "Executing Non-Inference Data Collection"
# # ################################################## NON-INFERENCE ##############################
# echo "Running EARCH MCNC"
# for arch_dir in "$arch_dir2";
# do
#     for b in $MCNC;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $EARCH $b $ARGSR "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done


# echo "Running STRATXIV MCNC"
# for arch_dir in "$arch_dir1";
# do
#     for b in $MCNC;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $STRATXIV $b $ARGSR "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done


# echo "Running STRATXIV TITAN"
# for arch_dir in "$arch_dir1";
# do
#     for b in $TITAN;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $STRATXIV $b $ARGSR "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done

# echo "Running STRATXIV TITANJR"
# for arch_dir in "$arch_dir1";
# do
#     for b in $TITANJR;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $STRATXIV $b $ARGSR "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done

###############################################################################################


################################################ INFERENCE ##############################
echo "Running EARCH MCNC"
for arch_dir in "$arch_dir2";
do
    for b in $MCNC;
    do  
        IFS="/." read -a bench <<< "$b"
        curdir=$arch_dir"_"${bench[-2]}
        mkdir -p $curdir
        mkdir -p $curdir/inference/
        mkdir -p $curdir/inference/processed/
        mkdir -p $curdir/inference/raw/
        mkdir -p $curdir/inference/combined/
        # echo $b
        # echo "$INITIAL $EARCH $b $ARGS "
        cd $curdir
        _mydir="$(pwd)"
        rm -f "$_mydir"/inference/*.csv
        rm -f "$_mydir"/inference/raw/*.csv
        eval "$INITIAL $EARCH $b $ARGS "
        cd /mnt/e/benchmarks/Outputs/

    done
done

# echo "Running STRATXIV MCNC"
# for arch_dir in "$arch_dir1";
# do
#     for b in $MCNC;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         _mydir="$(pwd)"
#         rm -f "$_mydir"/inference/*.csv
#         rm -f "$_mydir"/inference/raw/*.csv
#         eval "$INITIAL $STRATXIV $b $ARGS "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done
# echo "Running STRATXIV TITANJR"
# for arch_dir in "$arch_dir1";
# do
#     for b in $TITANJR;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         _mydir="$(pwd)"
#         rm "$_mydir"/inference/*.csv
#         rm "$_mydir"/inference/raw/*.csv
#         eval "$INITIAL $STRATXIV $b $ARGS "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done

# echo "Running STRATXIV TITANJR"
# for arch_dir in "$arch_dir1";
# do
#     for b in $TITANJR;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $STRATXIV $b $ARGS "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done

# echo "Running STRATXIV TITANJR"
# for arch_dir in "$arch_dir1";
# do
#     for b in $TITANJR;
#     do  
#         IFS="/." read -a bench <<< "$b"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#         # echo $b
#         # echo "$INITIAL $EARCH $b $ARGS "
#         cd $curdir
#         eval "$INITIAL $STRATXIV $b $ARGSR "
#         cd /mnt/e/benchmarks/Outputs/

#     done
# done

# # STRATXIV MCNC
# echo "Running STRATXIV MCNC"

# for arch_dir in "$arch_dir1";
# do
#     for f in $MCNC;
#     do
#         IFS="/." read -a bench <<< "$f"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#     done
# done

# # STRATXIV TITANJR
# echo "Running STRATXIV TITANJR"
# for arch_dir in "$arch_dir1";
# do
#     for f in $TITANJR;
#     do
#         IFS="/." read -a bench <<< "$f"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#     done
# done

# # STRATXIV TITAN
# echo "Running STRATXIV TITAN"

# for arch_dir in "$arch_dir1";
# do
#     for f in $TITAN,;
#     do
#         IFS="/." read -a bench <<< "$f"
#         curdir=$arch_dir"_"${bench[-2]}
#         mkdir -p $curdir
#         mkdir -p $curdir/inference/
#         mkdir -p $curdir/inference/processed/
#         mkdir -p $curdir/inference/raw/
#         mkdir -p $curdir/inference/combined/
#     done
# done