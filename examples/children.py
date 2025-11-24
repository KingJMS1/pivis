from pivis import Track, SupervisedAreasOfInterest
import matplotlib.pyplot as plt
import os
import pandas as pd
from multiprocessing import Pool



def run_file(filename, cluster_len, savedir, indir, num_top = 5):
    print(filename)
    track = Track.from_excel(os.path.join(indir, filename), 
                            20, 
                            "./IMG_Test.jpeg", 
                            "Recording timestamp", 
                            "Assisted mapping gaze point X [IMG_Test]", 
                            "Assisted mapping gaze point Y [IMG_Test]", 
                            "Eye movement type", 
                            "Fixation point X", 
                            "Fixation point Y")

    # The first time you run with supervised aois, you need to set the first_run flag to true.
    # This will create a folder called aois in your working directory, and output all aois found in the image into that directory.
    # Check these for anything that doesn't seem right, and ensure that the image does not contain feathering or blending artifacts.
    sup = SupervisedAreasOfInterest.from_labelled_img("AOIS_New_Test.png", track, cluster_len = cluster_len, first_run=False, remove_background = True)

    patterns, scores, interpret_scores, aoi_count, n_aois = sup.compute_patterns(5, [0, 1, 1, 1])
    _, children, _, _, _ = sup.compute_patterns(5, [0, 1, 1, 1], normalize="children")
    
    tocsv = pd.DataFrame.from_dict(children, orient="index")
    
    tocsv = tocsv.explode(["subpatterns", "scores"]).rename(columns={"scores": "subpattern_score"})

    return patterns[:num_top], scores[:num_top], interpret_scores[:num_top], aoi_count, n_aois, tocsv


def main():
    indir = "spread3"
    outdir = "output2"

    num_files_at_once = 2
    pool = Pool(num_files_at_once)
    
    # Can try a few different thresholds at once, to convert thresholds to time, just multiply by 20 msec for this project
    cluster_lens = [40, 60, 100]
    # 800, 1200, 2000

    first = False

    for cluster_len in cluster_lens:
        suffix = f"cl_{cluster_len}"
        savedir = os.path.join(outdir, suffix)
        os.makedirs(savedir, exist_ok=True)

        filenames = os.listdir(indir)

        label1 = []
        label2 = []
        label3 = []
        label4 = []
        label5 = []

        score1 = []
        score2 = []
        score3 = []
        score4 = []
        score5 = []

        iscore1 = []
        iscore2 = []
        iscore3 = []
        iscore4 = []
        iscore5 = []

        aoinums = []

        aoi_counts = {}

        print(f"Running multiple files, length: {cluster_len}")
        output = pool.starmap(run_file, [(filename, cluster_len, savedir, indir) for filename in filenames])
        print("Collecting output")
        for i, (patterns, scores, interpret_scores, aoi_count, n_aois, tocsv) in enumerate(output):
            tocsv.to_csv(savedir + "/" + filenames[i] + ".csv")
            s1, s2, s3, s4, s5 = scores
            i1, i2, i3, i4, i5 = interpret_scores
            l1, l2, l3, l4, l5 = [[int(y.item()) for y in x[0]] for x in patterns]
            
            
            score1.append(s1)
            score2.append(s2)
            score3.append(s3)
            score4.append(s4)
            score5.append(s5)

            iscore1.append(i1)
            iscore2.append(i2)
            iscore3.append(i3)
            iscore4.append(i4)
            iscore5.append(i5)
            
            label1.append(l1)
            label2.append(l2)
            label3.append(l3)
            label4.append(l4)
            label5.append(l5)
            aoinums.append(n_aois)

            for x in aoi_count:
                if x in aoi_counts:
                    aoi_counts[x].append(aoi_count[x])
                else:
                    aoi_counts[x] = [aoi_count[x]]

            

        pd.DataFrame({
            "File": filenames, 
            "Label1": label1, "Label2": label2, "Label3": label3, "Label4": label4, "Label5": label5, 
            "Score1": score1, "Score2": score2, "Score3": score3, "Score4": score4, "Score5": score5,
            "IScore1": iscore1, "IScore2": iscore2, "IScore3": iscore3, "IScore4": iscore4, "IScore5": iscore5,
            "Total_AOIs_Observed": aoinums, **aoi_counts
        }).to_csv(os.path.join(savedir, "main.csv"))
        
        plt.close("all")

if __name__ == "__main__":
    main()