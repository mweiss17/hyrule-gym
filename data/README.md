# hyrule-data
Data pre-processing for hyrule-gym

# Preprocessing steps

1. Extracting images from the Vuze+'s raw video with ffmpeg (8 cameras)
2. Crop those images to get each view
3. Undistort images 1 and 2 for orbslam (mono or stereo)
4. Stitch a panorama for each frame with Hugin panorama
