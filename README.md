# GNN-Accelerated FPGA Routing
This project is uses a Graph Neural Network to attempt to accelerate Pathfinder Routing of FPGAs. We used the MCNC & Titan Benchmarks using the VPR Example Architecture & Stratix IV. 

The current state of the project is negative results. The GNN currently both takes longer, more iterations, with worse Critical-Path-Delay. 

## Modified VPR
VTR to VPR is an Open Source tool. We modify it in order to both collect routing resource graph data for training and inference. On fixed minimum route width runs, we also modified it to allow batch collection of routing metrics. 

## Model
Our model uses PyTorch & PyTorch Geometric libraries.. We use TAGConv from PyTorch Geometric for our model. We use neighborhood sampling to split our graphs into subgraphs using neighborhood sampling for training. Subgraphs from the same graph are trained together. Each graph represents a single iteration of a routing resource graph. (Example: Iteration 1 of Tseng Circuit on StratixIV Architecture will get split into subgraphs & trained together).

Considering the size of graphs - the largest ones being 10 million nodes & 100 million edges - neighborhood sampling was the existing path forward to get the graphs to fit into GPU memory & train. However training a single model that takes in multiple large graphs & splits each into smaller subgraphs seems to be unexplored usage of the libraries for sampling. Using a subset of graphs that could fit on GPU memory (mainly Example Architecture with MCNC Benchmarks), neighborhood samping in this manner negatively affected results. However the GNN still showed no improvement to routing in either case. 

## Results:

Our Model resulted in generally worse performance of routing across the board. 

(PUT GRAPHS HERE)

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

