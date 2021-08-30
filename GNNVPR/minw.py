from optparse import OptionParser
import pandas as pd
def main(options):
    # This is a helper python file to allow us to dynamically set the route_channel_width.
    # For batch running of benchmarks. Defaults to using "minw.csv" in the current directory.
    # It simply prints the number to the screen corresponding to the input.
    # This runs & exits BEFORE VPR is ran, to avoid GIL conflict. 
    def helper(filename):
        return filename.split('/')[-1].split('.')[0]

    def even(x):
        if x % 2 == 0:
            return x
        else:
            return x + 1

    a_key = helper(options.arch).strip()
    c_key = helper(options.circuit).strip()
    df = pd.read_csv("minw.csv")
    df = df.set_index(['architecture', 'circuit'])
    ac_key  = even(int(df.loc[(a_key, c_key)]['minw']*1.20))

    # The Collection Shell Script will capture this output:
    print(ac_key)
    # print(df[a_key])

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--architectureFile", dest="arch",
                        help="Architecture File being used",
                        metavar="INPUT")
    parser.add_option("-c", "--circuitFile", dest="circuit",
                        help="Circuit BLIF being used.",
                        metavar="INPUT")
    (options, args) = parser.parse_args()
    main(options)   