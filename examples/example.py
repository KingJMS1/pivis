from pivis import Track, UnsupervisedAreasOfInterest
import matplotlib.pyplot as plt

track = Track.from_excel("/home/mascott/work/Eye Tracking/5 Minute Demo file (1).xlsx", 
                         20, 
                         "/home/mascott/work/Eye Tracking/IMG_4702.jpeg", 
                         "Recording timestamp", 
                         "Assisted mapping gaze point X [IMG_4702]", 
                         "Assisted mapping gaze point Y [IMG_4702]", 
                         "Eye movement type", 
                         "Fixation point X", 
                         "Fixation point Y")

aois = UnsupervisedAreasOfInterest.from_track(track, threshold=9, det_lim=5e7)

aois.plot(method="ovals", plot_transitions=False)
plt.savefig("out1.png")

aois.video(verbose=True)