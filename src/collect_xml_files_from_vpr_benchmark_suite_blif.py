import glob
import os
import sys
import subprocess
import logging
import multiprocessing as mp

command = dict()
command['initial'] = "$VTR_ROOT/vpr/vpr"
# command['args'] = "--route_chan_width 300 -j 6 --write_rr_graph" 
command['args'] = "--route_chan_width 300 -j 6"

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def parallel_test(arch, directory):
    directory = "/mnt/e"+directory
    arch = "/mnt/e"+arch
    pool = mp.Pool(mp.cpu_count())
    benchmarks = glob.glob(os.path.join(directory, '*.blif'))
    [pool.apply(collect_xml_rr_graph_per_directory, args = (b, arch, directory)) for b in benchmarks]
    pool.close()

def collect_xml_rr_graph_per_directory(b, arch, directory):
    arch_name = arch.split('/')[-1].split('.')[0]
    # print(arch_name)
    # print(arch)
    # print(directory)
    # # Get the base
    # # get archfile and directoryfile name
    
    bench_name = b.split('/')[-1].split('.')[0]
    version =  bench_name+"_"+arch_name
    # Changed for Titan Collection 
    # xmlfile = "../../XML/"+version+".xml"
    xmlfile = ""
    b_call = ""+command['initial']+" "+ arch+" "+b+" "+command['args'] +" " + xmlfile
    b_cwd = '/mnt/e/benchmarks'+"/Outputs/"+version+"/"
    ensure_dir(b_cwd)
    logging.debug(version+" started.")
    subprocess.run(b_call,cwd=b_cwd,shell=True,capture_output=True,bufsize=1_000_000_000)
    
    logging.debug(version+" executed & output:")
    # return outputs



def main():
    Earch = "/benchmarks/arch/MCNC/EArch.xml"
    # titan ="/benchmarks/arch/TITAN/stratixiv_arch.timing.xml"
    # MCNC = "/benchmarks/data/arch"
    # collect_xml_rr_graph_per_directory(titan,"/benchmarks/blif/TITAN/")
    # parallel_test(Earch,"/benchmarks/blif/MCNC/")
    parallel_test(Earch,"/benchmarks/blif/MCNC/test/")
    # parallel_test(titan,"/benchmarks/blif/TITAN/")
    # parallel_test(titan,"/benchmarks/blif/TITANJR/")


      
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # calling main function
    main()