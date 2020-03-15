import os
import os.path as osp
import shutil
import pandas as pd
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--summary_name', type=str, default='summary', help='what is the name of the file that contains the results saved')

# Note for later: can "generalize" the code by replacing hard coded names by variables
# also possible to add parser and parameters for this script (adapt code to other datasets etc)

if __name__ == '__main__':
    with open('objs.pkl', 'wb') as f:  # Just to create an "objs.pkl" object, so I don't get an error
        pickle.dump([0.,0.,''], f)
    model_one = "2020-02-21-04-5211"
    tmp_exp = "tmp_exp"

    args = parser.parse_args()
    summary_name = args.summary_name + ".csv"
    if(osp.exists(summary_name)):
        raise Exception('This file name (the results container) already exists')
    test_path = 'data/DRIVE/test.csv'
    df = pd.read_csv(test_path)
    im_list = df.im_paths
    k = 0
    im_nbs = []
    for k in range(len(im_list)):
        im_nb = im_list[k].split('_')[0]
        im_nb = im_nb.split('\\')[-1]
        im_nbs.append(im_nb)

    im_names = []
    hidden_scrs = []
    best_ims = []
    overf_best_scrs = []
    thresholds = [0.09,0.18,0.27,0.36,0.45,0.54,0.63,0.72,0.81,0.90]
    thresholds = [0.90]
    j = 0
    for im_name in im_nbs:
        for thr in thresholds:
            im_full = im_name + "_threshold_" + str(thr)
            im_names.append(im_full)
            print("-------------- Running the whole process for the image number: " + im_full + " --- (" + str(j) + "/" + str(len(im_nbs)-1) + ")")

            os.system("python image_selection.py --im " + im_name)

            os.system("python generate_results.py --so --exp " + model_one + " --forced_threshold " + str(thr))

            os.system("python analyze_results_by_img.py --so")

            with open('objs.pkl', 'rb') as f:
                hidden_score = pickle.load(f)[0]
            hidden_scrs.append(hidden_score)

            os.system("python train.py --so --forced_exp_name " + tmp_exp)

            os.system("python generate_results.py --train --exp " + tmp_exp)

            os.system("python analyze_results_by_img.py --train --exp " + tmp_exp)

            with open('objs.pkl', 'rb') as f:
                pickle_obj = pickle.load(f)
            overfitted_score, best_image = pickle_obj[1], pickle_obj[2]
            overf_best_scrs.append(overfitted_score)
            best_ims.append(best_image)
            # deleting the tmp_exp folder doesnt seem necessary for the moment
            print("-------------- hidden score: " + str(hidden_score) + " / overfitted score: " + str(overfitted_score) + " / highest score image: " + best_image)

            if osp.exists(summary_name):
                os.remove(summary_name)

            perf_df_test = pd.DataFrame({'im_nbs': im_names,
                                         'hidden_scrs': hidden_scrs,
                                         'overf_best_scrs': overf_best_scrs,
                                         'best_ims':best_ims})
            perf_df_test.to_csv(summary_name , index=False)
            print("--------------- Results saved to " + summary_name)
        j += 1


    # f= open("summary.txt","w+")
    # f.write("------------------------------------------------ SUMMARY, " + str(len(im_nbs)) + " images tested --------------------------------------------------\n\n")
    # i = 0
    # for im_name in im_nbs:
    #         f.write("image tested: " + im_name + " / hidden score: " + str(hidden_scrs[i]) + " / overfitted score: " + str(overf_best_scrs[i]) + " / highest score image: " + best_ims[i] + "\n\n")
    #         i += 1
