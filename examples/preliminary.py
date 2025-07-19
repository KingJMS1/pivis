from pivis import Track, SupervisedAreasOfInterest
import matplotlib.pyplot as plt
import os
import pandas as pd

indir = "newspread"
outdir = "output"

os.makedirs(outdir, exist_ok=True)

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

for filename in filenames:
    print(filename)
    track = Track.from_excel(os.path.join(indir, filename), 
                            20, 
                            "./NEW tower Image Raw.jpeg", 
                            "Recording timestamp", 
                            "Assisted mapping gaze point X [NEW tower Image]", 
                            "Assisted mapping gaze point Y [NEW tower Image]", 
                            "Eye movement type", 
                            "Fixation point X", 
                            "Fixation point Y")

    # The first time you run with supervised aois, you need to set the first_run flag to true.
    # This will create a folder called aois in your working directory, and output all aois found in the image into that directory.
    # Check these for anything that doesn't seem right, and ensure that the image does not contain feathering or blending artifacts.
    print("Reading Sup Aois from image")
    sup = SupervisedAreasOfInterest.from_labelled_img("Updated AOIS.png", track, first_run=False, remove_background = True)

    patterns, scores = sup.compute_patterns(5, [0, 1, 1, 1])
    s1, s2, s3, s4, s5 = scores[:5]
    l1, l2, l3, l4, l5 = [x[0] for x in patterns][:5]
    
    score1.append(s1)
    score2.append(s2)
    score3.append(s3)
    score4.append(s4)
    score5.append(s5)
    
    label1.append(l1)
    label2.append(l2)
    label3.append(l3)
    label4.append(l4)
    label5.append(l5)


    print("Caching")
    sup.save(os.path.join(outdir, filename + ".pkl"))

pd.DataFrame({"File": filenames, "Label1": label1, "Label2": label2, "Label3": label3, "Label4": label4, "Label5": label5, "Score1": score1, "Score2": score2, "Score3": score3, "Score4": score4, "Score5": score5}).to_csv(os.path.join(outdir, "main.csv"))