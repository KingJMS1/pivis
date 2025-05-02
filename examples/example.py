from pivis_mascottpy import Track, AOIS
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

aois = AOIS.from_track(track)

aois.plot()
plt.savefig("out1.png")

aois.video()