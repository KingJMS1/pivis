from pivis import Track, AreasOfInterest
import matplotlib.pyplot as plt
import datetime
import time
import os
import shutil
from datetime import datetime

# --- PATH CONFIGURATION ---
input_folder = "C:/Users/David/Documents/VSCODE/COA_Eyetracking/pivis/examples/Files pending to be processed"
output_folder = "C:/Users/David/Documents/VSCODE/COA_Eyetracking/pivis/examples/Files processed"
os.makedirs(output_folder, exist_ok=True)

# Ask the user for filenames
excel_base = input("Enter the Excel filename: ").strip()
image_base = input("Enter the image filename: ").strip()
# Add extensions
excel_file = f"{excel_base}.xlsx"
# Full path to Excel file
excel_path = os.path.join(input_folder, excel_file)
# Try common image extensions
image_extensions = [".jpg", ".jpeg", ".png"]
image_file = None

image_path = None
for ext in image_extensions:
    candidate = image_base + ext
    test_path = os.path.join(input_folder, candidate)
    if os.path.exists(test_path):
        image_file = candidate
        image_path = test_path
        break
image_name_no_ext = os.path.splitext(image_file)[0]

gaze_x_col = f"Assisted mapping gaze point X [{image_name_no_ext}]"
gaze_y_col = f"Assisted mapping gaze point Y [{image_name_no_ext}]"

# --- FIND EXCEL AND IMAGE FILE ---
#excel_file = next((f for f in os.listdir(input_folder) if f.endswith(".xlsx")), None)
#image_file = next((f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))), None)
# Validate files
if not os.path.exists(excel_path):
    raise FileNotFoundError(f"Excel file not found: {excel_path}")
if not image_path:
    raise FileNotFoundError(f"Image file not found for base name: {image_base}")

file_label = f"{excel_file} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Start the timer
start_time = time.perf_counter()

track = Track.from_excel(excel_path, 
                         20, 
                         image_path, 
                         "Recording timestamp", 
                         gaze_x_col, 
                         gaze_y_col, 
                         "Eye movement type",
                         "Fixation point X",
                         "Fixation point Y")

#Generate AOIs
aois = AreasOfInterest.from_track(track, threshold=9, det_lim=5e7)

#Plot and save
basename = excel_base
output_img = os.path.join(output_folder, f"{basename}_output.png")
aois.plot(method="ovals", plot_transitions=True) #Plot_transitions enable True
plt.savefig(output_img)

#Save video in processed path
output_vid = os.path.join(output_folder, f"{basename}.mp4")
aois.video(fileloc=output_vid, verbose=True, label=file_label)

#Move Excel file to processed folder
shutil.move(excel_path, os.path.join(output_folder, excel_file))
shutil.move(image_path, os.path.join(output_folder, image_file))

# End the timer
end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Format and print results
finish_timestamp = datetime.now().strftime("%H:%M:%S")
print(f"✅ Program finished at {finish_timestamp}.")
print(f"🕒 Total execution time: {elapsed_time:.2f} seconds.")