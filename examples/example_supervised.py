from pivis import Track, SupervisedAreasOfInterest
import matplotlib.pyplot as plt

print("Reading Track")
track = Track.from_excel("213 S4.xlsx", 
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
sup = SupervisedAreasOfInterest.from_labelled_img("NEW tower Image AOIs.png", track, first_run = True)
print("Plotting")
sup.plot()
plt.savefig("example_img.png")
print("Video")
sup.video()
print("Caching")
sup.save()