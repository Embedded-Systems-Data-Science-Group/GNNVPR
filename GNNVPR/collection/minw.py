from optparse import OptionParser
import pandas as pd
def main(options):
    def helper(filename):
        # Same Behavior as C Program splitting.
        return filename.split('/')[-1].split('.')[0]


    a_key = helper(options.arch).strip()
    c_key = helper(options.circuit).strip()
    df = pd.read_csv("/benchmarks/GNNVPR/GNNVPR/collection/minw.csv")
    df = df.set_index(['architecture', 'circuit'])
    print(int(df.loc[(a_key, c_key)]['minw']*1.20))
    # print(df[a_key])

    # Eventually we will just print the number
    # print(a_key, " ", c_key)


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