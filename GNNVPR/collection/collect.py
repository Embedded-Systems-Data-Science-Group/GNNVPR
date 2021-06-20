"""Collects Data from VPR for the GNN
"""
import glob
import logging
import multiprocessing as mp
import os
import subprocess


command = dict()
command['initial'] = "$VTR_ROOT/vpr/vpr"
command['args'] = "--route_chan_width 300 -j 6 --collect_data on "\
    "--do_inference on"


def ensure_dir(file_path):
    """Function checks if directory exists & creates it if not.
    Parameters
    ----------
    file_path : string
            passed in directory path.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def parallel_collect(arch, directory):
    """This function attempts a parallelization of the data collection,
    it calls collect_per_directory using pool.apply.

    I don't this function works correctly, especially with IO &
    the interpreter lock.

    Parameters
    ----------
    arch : string
       file path to architecture blif file.
    directory : string
        file path to the directory containing the set of benchmarks to run.
    """
    directory = "/mnt/e"+directory
    arch = "/mnt/e"+arch
    pool = mp.Pool(mp.cpu_count())
    benchmarks = glob.glob(os.path.join(directory, '*.blif'))
    [pool.apply(collect_per_directory, args=(
        b, arch, directory)) for b in benchmarks]
    pool.close()


def collect_per_directory(b, arch, directory):
    """For a given benchmark, execute the VTR-VPR Suite using subprocess.

    Function Paramters should be updated, but sets the working directory &
    call arguments using a global definition

    Parameters
    ----------
    b : string
        specific benchmark path given in the directory.
    arch : string
        file path to architecture blif file.
    directory : string
       file path to the directory containing the set of benchmarks
       - unused here.
    """
    arch_name = arch.split('/')[-1].split('.')[0]
    bench_name = b.split('/')[-1].split('.')[0]
    version = bench_name+"_"+arch_name
    # Changed for Titan Collection

    b_call = ""+command['initial']+" " + arch + \
        " "+b+" "+command['args'] + " "
    b_cwd = '/mnt/e/benchmarks'+"/Outputs/"+version+"/"
    ensure_dir(b_cwd)
    logging.debug(version+" started.")
    subprocess.run(b_call, cwd=b_cwd, shell=True,
                   capture_output=True, bufsize=1_000_000_000)

    logging.debug(version+" executed & output:")
    # return outputs


def main():
    """This function collects batch runs a set of benchmarks.
    VPR handles all the output of files

    Should be updated to work based on input, at least from some options.
    """
    Earch = "/benchmarks/arch/MCNC/EArch.xml"

    # titan ="/benchmarks/arch/TITAN/stratixiv_arch.timing.xml"
    # MCNC = "/benchmarks/data/arch"
    # collect_per_directory(titan,"/benchmarks/blif/TITAN/")
    # parallel_test(Earch,"/benchmarks/blif/MCNC/")
    parallel_collect(Earch, "/benchmarks/blif/MCNC/test/")
    # parallel_test(titan,"/benchmarks/blif/TITAN/")
    # parallel_test(titan,"/benchmarks/blif/TITANJR/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # calling main function
    main()
