
root_folder_all_data = "/home/crosscdr"

splits_root_folder = "/home/jovyan/Torte/splits/"

enable_graph = True # is set to True for the Graph version
enable_global_graph_features = True
plotImg = False
invert = False
dict_root = "/home/jovyan/CrossCDR/"

enableVideo = False
enable_SUM = True

if enable_SUM:
    resolution = (256, 256) # sum resolution

transform = True
debug = False

###### json parser settings
outer_tag = 'FrameNumber'
nested_tag_path_gazeOrigin = ['Participant', 'Eyes', 'gazeOrigin']
nested_tag_path_pos3d = ['Participant', 'Eyes', 'pos3d']
#######

epochs = 15

#training settings
batch=32
shuffle = True

phase = "train"

trials = ["Baseline", "H_R_1-1", "H_R_1-2", "H_R_2-1", "H_S_1-1", "H_S_1-2", "H_S_2-1", "Zebra"]

urls_generated = True

#output_missing_file_path = splits_root_folder+"missing_folders.txt"  # File to store valid URLs

individual_files_are_listed = True