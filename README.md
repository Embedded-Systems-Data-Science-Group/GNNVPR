# GNN-Accelerated FPGA Routing
This project uses a Graph Neural Network to attempt to accelerate Pathfinder Routing of FPGAs. We used the MCNC & Titan Benchmarks using the VPR Example Architecture & Stratix IV. 

The current state of the project is negative results. The GNN currently both takes longer, more iterations, with worse Critical-Path-Delay. 

## Modified VPR
VTR to VPR is an Open Source tool. We modify it in order to both collect routing resource graph data for training and inference. On fixed minimum route width runs, we also modified it to allow batch collection of routing metrics. 

## Model
Our model uses PyTorch & PyTorch Geometric libraries.. We use TAGConv from PyTorch Geometric for our model. We use neighborhood sampling to split our graphs into subgraphs using neighborhood sampling for training. Subgraphs from the same graph are trained together. Each graph represents a single iteration of a routing resource graph. (Example: Iteration 1 of Tseng Circuit on StratixIV Architecture will get split into subgraphs & trained together).

Considering the size of graphs - the largest ones being 10 million nodes & 100 million edges - neighborhood sampling was the existing path forward to get the graphs to fit into GPU memory & train. However training a single model that takes in multiple large graphs & splits each into smaller subgraphs seems to be unexplored usage of the libraries for sampling. Using a subset of graphs that could fit on GPU memory (mainly Example Architecture with MCNC Benchmarks), neighborhood samping in this manner negatively affected results. However the GNN still showed no improvement to routing in either case. 

## Results:

Our Model resulted in generally worse performance of routing across the board: (init is the GNN)
```
Average:  -9.27%
                 init  reg time_init time_reg CPD_init CPD_REG Reduction
EArch__alu4        22   18      5.34     0.43     8.42    5.24   -22.22%
EArch__apex2       19   16     13.45     0.60     8.75    6.22   -18.75%
EArch__apex4       22   21     19.90     0.55     9.59    5.50    -4.76%
EArch__bigkey      18   15      0.75     0.40     3.78    3.11   -20.00%
EArch__clma        19   16    122.95     2.45    13.39   11.06   -18.75%
EArch__des         17   15      6.24     0.54     7.66    5.83   -13.33%
EArch__diffeq      15   15      1.79     0.31     7.82    6.56     0.00%
EArch__dsip        15   17      1.82     0.41     3.88    2.77    11.76%
EArch__elliptic    19   17     37.26     1.00    10.13    9.33   -11.76%
EArch__ex1010      19   17     34.34     1.44    10.08    7.57   -11.76%
EArch__ex5p        19   19      9.31     0.40     8.52    5.30     0.00%
EArch__frisc       15   14     16.74     0.92    12.64   12.50    -7.14%
EArch__misex3      20   18      9.02     0.47     8.68    5.48   -11.11%
EArch__pdc         20   18    218.36     1.92    13.18    8.01   -11.11%
EArch__s298        18   18      1.91     0.45    12.05   10.13     0.00%
EArch__s38417      17   16      5.88     1.20    10.28    7.84    -6.25%
EArch__s38584      18   16     11.80     1.13     8.54    6.58   -12.50%
EArch__seq         20   18     18.75     0.60     8.60    5.24   -11.11%
EArch__spla        22   20    118.15     1.64    11.46    7.22   -10.00%
EArch__tseng       16   15      0.56     0.22     8.22    6.47    -6.67%
```

## How to Run:
We assume you are running on Linux, with PyTorch & PyTorch Geometric running & working on a GPU. 
Generally the flow of the programs are as follows:
1. VTR Verilog to Routing
    1. Initialize submodule to run install forked vpr-to-vtr (RoutingCollection branch)
    2. Configure $VTR_ROOT in shell
    3. make in vtr-directory (use -j to make it parallel - takes a while the first time)
2. Configure folders:
    We assume the following structure:
    ```
    benchmarks/
        arch/
            MCNC/
            TITAN/
        blif/
            MCNC/
            TITAN/
            TITANJR/
        GNNVPR/ (Github Repository)
            GNNVPR/
                readme.md (YOU ARE HERE)
            submodules/
                vtr-verilog-to-routing/
        graph_data/
        Outputs/
            inf.sh that contains:
                "_mydir="$(pwd)
                python /benchmarks/GNNVPR/GNNVPR/training/inference.py -i $_mydir/inference/"
        route_metrics/
        training/
            raw/
            processed/
            combined/
        

    ```
3. Running Collection, In collect.sh, set "BASEDIRECTORY" to the /benchmarks folder listed above, notice how everything follows from that. The function "run_benchmark" takes 5 arguments and currently uses dynamic route_channel_width of minw:
    1. Print Statement to indicate what is being run
    2. the architecture directory (for splitting outputs into subfolders )
    3. A folder containing blif files to be ran
    4. The actual architecture file
    5. Arguments for VPR. (See # Arguments section of folder). #Ignore Output lets you change ERR to allow ignoring VPR output - useful for when running many benchmarks. 


Step 3 can output graph data for training, inference from a trained model, and/or collect route metrics. It can also run standard VPR runs in a batched format.

4. Model Training. 
    1. Copy output graphs collected from VPR that are in graph_data/ into training/ & recopy into raw/ (this is a bug I havent fixed yet, required redundant copying)
    2. Run the following command:
    ```
    python GNNVPR/model.py -I benchmarks/training/ -O Outputs/ -r Outputs/
    ```
    3. Processing should commence & finish, giving you an idea of the progress.
    4. Training Should then start, it prints the 1st epoch & then every 10th epoch after. You can adjust this in model.py
5. Model Inference. 
    1. Assumes a model.pt is output from step 4 and inf.sh is in Outputs/. 
    2. Batch run collect.sh with the --gnntype 2 as an argument, this lets the modified VPR fork know to inference run. 

