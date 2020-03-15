import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--summary_name', type=str, default='summary', help='what is the name of the file that contains the results saved')

if __name__ == '__main__':

    args = parser.parse_args()
    summary_name = args.summary_name + ".csv"

    h_scs = [
    0.8204322926288564,
    0.8518381305215161,
    0.8096403342721271,
    0.8370941384262393,
    0.8246467377862784,
    0.8161733649566263,
    0.8281887408213218,
    0.8157353566869947,
    0.8206554419723392,
    0.8149520180637877,
    0.8061295042691065,
    0.8295131559264,
    0.8276779163609684,
    0.8220483862612029,
    0.81196333815726,
    0.8390248652859833,
    0.8104725369915322,
    0.8075678452187313,
    0.8287214736537813,
    0.8000143256213739]

    o_scs = [
    0.7858975079106517,
    0.7960585249328158,
    0.783943371394017,
    0.8184248324289818,
    0.7950124057714494,
    0.8115147627655646,
    0.8122286476066944,
    0.7972025201093632,
    0.8182557259398391,
    0.820864312267658,
    0.8165927030048443,
    0.8140589108206452,
    0.8198349108962845,
    0.7986640305583815,
    0.7905500974807905,
    0.8268385584529739,
    0.8097343974878974,
    0.8140951565732221,
    0.8161865752291497,
    0.7983728998957426]

    df = pd.read_csv(summary_name)
    h_scs, o_scs = df.hidden_scrs.array, df.overf_best_scrs.array
    print(h_scs, o_scs)
    # test of perfect correlation
    # h_scs = np.asarray(range(20))
    # o_scs = 2 * h_scs

    # test of random correlation
    # h_scs = np.random.random(20,)
    # o_scs = np.random.random(20,)

    # Pearson correlation coefficient
    cc = np.corrcoef(h_scs,o_scs)
    plt.scatter(h_scs, o_scs, label = "Correlation Coefficient = " + str(cc[0][1])
    , color= "blue", marker= "*", s=30)

    # x-axis label
    plt.xlabel('Hidden Scores')
    # frequency label
    plt.ylabel('Overfitted Scores')
    # plot title
    plt.title('Scores Correlation')
    # showing legend
    plt.legend()
    # function to show the plot
    plt.show()




    # ---------------------------------------------------------------------------------------
    #
    # image tested: 01 / hidden score:  / overfitted score:  / highest score image: 32_training
    #
    # image tested: 02 / hidden score:  / overfitted score:  / highest score image: 24_training
    #
    # image tested: 03 / hidden score:  / overfitted score:  / highest score image: 26_training
    #
    # image tested: 04 / hidden score:  / overfitted score:  / highest score image: 21_training
    #
    # image tested: 05 / hidden score:  / overfitted score:  / highest score image: 35_training
    #
    # image tested: 06 / hidden score:  / overfitted score:  / highest score image: 21_training
    #
    # image tested: 07 / hidden score:  / overfitted score:  / highest score image: 35_training
    #
    # image tested: 08 / hidden score:  / overfitted score:  / highest score image: 21_training
    #
    # image tested: 09 / hidden score:  / overfitted score:  / highest score image: 24_training
    #
    # image tested: 10 / hidden score:  / overfitted score:  / highest score image: 35_training
    #
    # image tested: 11 / hidden score:  / overfitted score:  / highest score image: 35_training
    #
    # image tested: 12 / hidden score:  / overfitted score:  / highest score image: 21_training
    #
    # image tested: 13 / hidden score:  / overfitted score:  / highest score image: 21_training
    #
    # image tested: 14 / hidden score:  / overfitted score:  / highest score image: 21_training
    #
    # image tested: 15 / hidden score:  / overfitted score:  / highest score image: 26_training
    #
    # image tested: 16 / hidden score:  / overfitted score:  / highest score image: 24_training
    #
    # image tested: 17 / hidden score:  / overfitted score:  / highest score image: 24_training
    #
    # image tested: 18 / hidden score:  / overfitted score:  / highest score image: 24_training
    #
    # image tested: 19 / hidden score:  / overfitted score:  / highest score image: 36_training
    #
    # image tested: 20 / hidden score:  / overfitted score:  / highest score image: 35_training
